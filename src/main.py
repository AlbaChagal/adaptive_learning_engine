import os
import random
import numpy as np
import torch
from typing import Dict, List, Tuple

from src.misc.config import Config
from src.misc.data_structures import FeatureReport, PolicyReport, CoachingLog, CoachingCard
from src.evaluation.weakest_skill_first_policy import WeakestSkillFirst
from src.logging.result_reporter import ResultReporter
from src.misc.types import NumberType, ContextRowType
from src.inference.dataset_manager import DatasetManager, Session
from src.inference.features import FeaturesExtractor
from src.inference.llm_client import LLMClient
from src.evaluation.linucb_policy import LinUCB
from src.inference.coaching_card_generator import CoachingCardGenerator
from src.evaluation.evaluator import LosoEvaluator


class AdaptiveLearningEngine(object):
    """
    The main engine for adaptive learning with coaching cards
    """
    def __init__(self):
        self.config = Config()
        self._set_global_seed(self.config.random_seed)
        self._ensure_dir(self.config.artifacts_dir)

        # Init Inference Modules
        self.llm = LLMClient(config=self.config, is_debug=self.config.is_debug_main)
        self.feature_extractor = FeaturesExtractor(cache_path=self.config.feature_cache_path,
                                                   is_debug=self.config.is_debug_main,
                                                   seed=self.config.random_seed,
                                                   llm_callable=self.llm)
        self.coaching_generator: CoachingCardGenerator = CoachingCardGenerator(is_debug=self.config.is_debug_main,
                                                                               config=self.config)
        self.dataset_manager: DatasetManager = DatasetManager(config=self.config,
                                                              is_debug=self.config.is_debug_main)
        self.loso_evaluator: LosoEvaluator = LosoEvaluator(config=self.config,
                                                           is_debug=self.config.is_debug_main)
        self.reporter: ResultReporter = ResultReporter(config=self.config)

        self.sessions = self.dataset_manager.load_dataset(path=self.config.dataset_file_path,
                                                          is_create_synthetic_dataset=False)
        self.num_sessions: int = len(self.sessions)

        # Init evaluation Modules
        n_features: int = (2 * len(self.feature_extractor)) + 2  # baseline + llm + delta_overall + delta_skill_avg
        self.arms: List[str] = self.get_rubrics()
        self.bandit: LinUCB = LinUCB(config=self.config,
                                     n_features=n_features,
                                     arms=self.arms,
                                     is_debug=self.config.is_debug_main)
        self.weak_first_calculator = WeakestSkillFirst(config=self.config,
                                                       is_debug=self.config.is_debug_main)

    @staticmethod
    def get_rubrics() -> List[str]:
        """
        Return the action space for the policy, per spec.
        """
        return ["clarity", "active_listening", "call_to_action", "friendliness"]

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """
        Ensure a directory exists or create it if missing
        :param path: The path to the directory
        """
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        """
        Set the random seed for reproducibility across random, numpy, and torch.
        :param seed: The seed value to set
        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def compute_reward(self, curr_sess: Session, next_sess: Session, focus: str) -> float:
        """
        Compute the reward score
        :param curr_sess: The data from the current session
        :param next_sess: The data from the next session
        :focus: A string indicating the skill in focus
        :return: A reward score
        """
        d_overall: float = (next_sess.overall - curr_sess.overall) / 100.0
        d_focus: float = getattr(next_sess, focus) - getattr(curr_sess, focus)

        d_overall = max(-1.0, min(1.0, d_overall))
        d_focus = max(-1.0, min(1.0, d_focus))

        return (self.config.focus_factor * d_focus) + (self.config.overall_factor * d_overall)

    @classmethod
    def build_context_row(cls,
                          sess: Session,
                          baseline_features: Dict[str, NumberType],
                          llm_features: Dict[str, NumberType],
                          prev_sess: Session = None) -> ContextRowType:
        """
        A method to generate a context row dict from the
        session data, baseline feature, llm features and deltas
        :param sess: The session data
        :param baseline_features: The baseline features
        :param llm_features: The LLM features
        :param prev_sess: The previous session data to compute deltas
        :return: A dict containing all relevant data
        """
        row: ContextRowType = {
            "session_id": sess.session_id,
            "overall": sess.overall,
            "rubrics": {rub: sess.__getattribute__(rub) for rub in cls.get_rubrics()},
            "baseline_features": baseline_features,
            "llm_features": llm_features,
        }

        # improvement proxies if prev exists
        if prev_sess is not None:
            row["delta_overall"] = sess.overall - prev_sess.overall
            row["delta_skill_avg"] = ((sess.clarity + sess.active_listening + sess.call_to_action + sess.friendliness) / 4.0) - \
                                     ((prev_sess.clarity + prev_sess.active_listening + prev_sess.call_to_action + prev_sess.friendliness)/4.0)
        else:
            row["delta_overall"] = 0.0
            row["delta_skill_avg"] = 0.0
        return row

    def get_all_rows(self, is_save_csv: bool = True) -> List[ContextRowType]:
        """
        A method to load all data points (rows)
        :param is_save_csv: A boolean indicating if it should save the features or not
        :return: A list of all data points (rows)
        """
        rows: List[ContextRowType] = []
        baseline_features: ContextRowType
        llm_features: ContextRowType
        prev_sess: Session

        for i, sess in enumerate(self.sessions):
            baseline_features, llm_features = \
                self.feature_extractor.extract_features(sess.transcript)
            prev_sess = self.sessions[i - 1] if i > 0 else None
            rows.append(self.build_context_row(sess=sess,
                                               baseline_features=baseline_features,
                                               llm_features=llm_features,
                                               prev_sess=prev_sess))
        if is_save_csv:
            FeaturesExtractor.to_csv(rows, os.path.join(self.config.artifacts_dir, "features.csv"))
        return rows

    def get_feature_report(self, rows: List[ContextRowType]) -> FeatureReport:
        """
        Create a feature report
        :param rows: The data rows containing baseline_features, LLM_features and y_overall_labels
        :return: The feature report
        """
        # build matrices for evaluation targets (use steps sess_ind->sess_ind+1)
        X_base_list: List[List[NumberType]] = []
        X_llm_list: List[List[NumberType]] = []
        y_list: List[NumberType] = []

        curr_row: ContextRowType
        next_row: ContextRowType

        for sess_ind in range(len(rows) - 1):
            curr_row, next_row = rows[sess_ind: sess_ind + 2]
            X_base_list.append(
                [curr_row["baseline_features"][key]
                 for key in self.feature_extractor.get_baseline_feature_to_calc_func_dict().keys()]
            )
            X_llm_list.append(
                [curr_row["llm_features"][k]
                 for k in self.feature_extractor.get_baseline_feature_to_calc_func_dict().keys()]
            )
            y_list.append(next_row["overall"] - curr_row["overall"])

        X_base: np.ndarray = np.array(X_base_list, dtype=float) if X_base_list \
            else np.zeros((0, len(self.feature_extractor)))
        X_llm: np.ndarray = np.array(X_llm_list, dtype=float) if X_llm_list \
            else np.zeros(len(self.feature_extractor))
        y: np.ndarray = np.array(y_list, dtype=float) if y_list else np.zeros((0,))

        # Calculate LOSO evaluation
        feature_report: FeatureReport
        try:
            feature_report = self.loso_evaluator.eval_pair(X_base, X_llm, y)
        except ValueError as e:
            print(f"Warning: LOSO evaluation skipped: {e}")
            feature_report = FeatureReport(r2_baseline=0.0, r2_with_llm=0.0, delta=0.0)
        return feature_report

    def calculate_policies_rewards(
            self,
            rows: List[ContextRowType]
    ) -> Tuple[List[float], List[float], int, int, List[CoachingLog]]:
        """
        Calculate rewards according the policies and create a coaching log
        :param rows: All data points (rows)
        rewards_base, rewards_lin, pos_overall_base, pos_overall_lin, coaching_log

        :return: 1. Weakest skill first policy reward score
                 2. LinUCB policy reward score
                 3. The percentage of cases where weakest-skill-first policy has improved
                 4. The percentage of cases where LinUCB policy has improved
                 5. The coaching log
        """
        last_actions: List[str] = []
        coaching_log: List[CoachingLog] = []
        rewards_lin: List[float] = []
        rewards_base: List[float] = []
        pos_overall_lin: int = 0
        pos_overall_base: int = 0

        curr_row: ContextRowType
        x_t: np.ndarray
        choice: str
        rubric: Dict[str, NumberType]
        feats: Dict[str, NumberType]
        card: CoachingCard
        curr_session: Session
        next_session: Session
        reward: float
        base_focus: str
        base_reward: float

        for sess_ind in range(self.num_sessions - 1):
            curr_row = rows[sess_ind]
            x_t = self.feature_extractor.create_context_vector(curr_row)
            choice = self.bandit.select(x=x_t,
                                        last_actions=last_actions,
                                        max_repeat=self.config.linucb_max_focus_repeat)
            last_actions.append(choice)

            # coaching card for next session
            rubric = {k: curr_row["rubrics"][k] for k in self.arms}
            feats = {k: (self.config.baseline_feature_factor * curr_row["baseline_features"][k]) +
                        (self.config.llm_feature_factor * curr_row["llm_features"][k])
                     for k in self.feature_extractor.get_feature_names()}

            card = self.coaching_generator.generate(rubric=rubric, features=feats, focus=choice)
            coaching_log.append(CoachingLog(turn=sess_ind, focus=choice, card=card))

            # observe next session to compute reward
            curr_session, next_session = self.sessions[sess_ind: sess_ind + 2]
            reward = self.compute_reward(curr_sess=curr_session, next_sess=next_session, focus=choice)
            rewards_lin.append(reward)
            if (next_session.overall - curr_session.overall) > 0:
                pos_overall_lin += 1

            self.bandit.update(choice, x_t, reward)

            # baseline policy: weakest-skill-first
            base_focus = self.weak_first_calculator.select(self.sessions[sess_ind])
            base_reward = self.compute_reward(curr_sess=curr_session, next_sess=next_session, focus=base_focus)
            rewards_base.append(base_reward)
            if (next_session.overall - curr_session.overall) > 0:
                pos_overall_base += 1

        return rewards_base, rewards_lin, pos_overall_base, pos_overall_lin, coaching_log

    def run(self):
        """
        The main function / test for the entire adaptive learning engine pipeline
        """
        rows: List[ContextRowType] = self.get_all_rows()
        feature_report: FeatureReport = self.get_feature_report(rows=rows)

        rewards_base: List[float]
        rewards_lin: List[float]
        pos_overall_base: int
        pos_overall_lin: int
        rewards_base, rewards_lin, pos_overall_base, pos_overall_lin, coaching_log = \
            self.calculate_policies_rewards(rows=rows)

        self.coaching_generator.to_json(coaching_log=coaching_log,
                                        path=self.config.coaching_log_path)
        self.coaching_generator.log_coaching_next_from_coaching_log(coaching_log=coaching_log)

        policy_report: PolicyReport = (
            self.loso_evaluator.compare_policies(
                rewards_baseline=rewards_base,
                rewards_linucb=rewards_lin,
                pos_overall_base=self.loso_evaluator.calc_pos_overall_rate(pos_overall=pos_overall_base,
                                                                           num_sessions=self.num_sessions),
                pos_overall_lin=self.loso_evaluator.calc_pos_overall_rate(pos_overall=pos_overall_lin,
                                                                          num_sessions=self.num_sessions),
            )
        )

        self.reporter.generate_report(feature_report=feature_report,
                                      policy_report=policy_report,
                                      num_sessions=self.num_sessions,
                                      num_features=len(self.feature_extractor),
                                      action_keys=self.get_rubrics())

if __name__ == "__main__":
    adaptive_learning_engine = AdaptiveLearningEngine()
    adaptive_learning_engine.run()

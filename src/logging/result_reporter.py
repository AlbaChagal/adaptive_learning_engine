import ast
import csv
import json
from torch import NumberType
from typing import List, Dict, Any

from src.misc.config import Config
from src.misc.data_structures import PolicyReport, FeatureReport
from src.logging.logger import Logger


class ResultReporter(object):
    """
    A class to generate and save evaluation reports
    """
    def __init__(self, config: Config):
        """
        Initialize the result reporter
        :param config: The configuration object
        :return: None
        """
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,"info")

    def _alignment_examples(self, action_keys: List[str], k: int = 3) -> List[str]:
        """
        Try to build K alignment examples from artifacts:
        - Read artifacts/features.csv
        - Read artifacts/coaching_log.json
        - For each turn, compare chosen focus to the weakest rubric among the 4 actions
        Returns human-readable lines; safe if artifacts missing.
        """
        try:
            with open(self.config.coaching_log_path, "r",
                      encoding=self.config.encoding_protocol) as f:
                logs: Any = json.load(f)

            # 2) Load features.csv rows (we only need rubrics per step)
            #    Format is compact: columns include 'rubrics' as a dict-like string.
            rows: List[Dict[str, Any]] = []
            with open(self.config.features_csv_path, "r",
                      encoding=self.config.encoding_protocol, newline="") as cf:
                reader: csv.DictReader = csv.DictReader(cf)
                for r in reader:
                    rows.append(r)

            examples: List[str] = []
            turn: int
            focus: str
            rubrics_raw: str
            rubrics: Dict[str, NumberType]
            v: NumberType
            v_norm: float
            weakest_key: str
            weakest_val: float
            for log in logs:
                turn = log.get("turn", None)
                focus = log.get("focus", "")
                if turn is None or turn >= len(rows):
                    continue

                rubrics_raw = rows[turn].get("rubrics", "{}")
                try:
                    rubrics = ast.literal_eval(rubrics_raw)
                except Exception:
                    rubrics = {}

                weakest_key, weakest_val = None, None
                for kname in action_keys:
                    v = rubrics.get(kname, None)
                    if v is None:
                        continue
                    if isinstance(v, (int, float)) and v > 1.0:
                        v_norm = v / 100.0
                    else:
                        v_norm = float(v or 0.0)
                    if weakest_val is None or v_norm < weakest_val:
                        weakest_key, weakest_val = kname, v_norm

                if weakest_key is None:
                    continue

                match_str: str = "match" if focus == weakest_key else "mismatch"
                examples.append(
                    f"turn={turn}: focus={focus} | weakest={weakest_key} ({weakest_val:.2f}) -> {match_str}"
                )

                if len(examples) >= k:
                    break

            return examples

        except Exception as e:
            self.logger.warning(f'_alignment_examples - failed aligning samples with error: {e}')
            return []

    def generate_report(self,
                        feature_report: FeatureReport,
                        policy_report: PolicyReport,
                        num_sessions: int,
                        num_features: int,
                        action_keys: List[str]) -> None:
        """
        Generate and save the evaluation report
        :param feature_report: The feature ablation report
        :param policy_report: The policy comparison report
        :param num_sessions: The number of sessions evaluated
        :param num_features: The number of features used
        :param action_keys: A List containing all action keys
        :return: None
        """
        report_lines = []
        report_lines.append("")
        report_lines.append("=== Coaching Policy Evaluation Report ===")
        report_lines.append("")
        report_lines.append(f"Sessions: {num_sessions} | Steps: {num_sessions - 1}")
        report_lines.append(f"Features: baseline={num_features} "
                            f"LLM={num_features}")
        report_lines.append(f"Policy = LinUCB (alpha={self.config.linucb_alpha}) | "
                            f"Reward = {self.config.focus_factor}*delta(focus)+"
                            f"{self.config.overall_factor}*delta(overall)")
        report_lines.append("")
        report_lines.append("Feature ablation (LOSO):")
        report_lines.append(f"- R2 baseline: {feature_report.r2_baseline:.3f}")
        report_lines.append(f"- R2 with LLM: {feature_report.r2_with_llm:.3f}")
        report_lines.append(f"- R2 delta: {feature_report.delta:.3f}")
        report_lines.append("")
        report_lines.append("Policy comparison:")
        report_lines.append(
            f"- Weakest-skill-first: mean reward={policy_report.mean_rewards_baseline:.4f}, "
            f"overall_delta={policy_report.pos_overall_base:.2%}")
        report_lines.append(
            f"- LinUCB (+LLM feats): mean reward={policy_report.mean_rewards_linucb:.4f}, "
            f"overall_delta={policy_report.pos_overall_linucb:.2%}")

        examples = self._alignment_examples(k=3, action_keys=action_keys)
        if examples:
            report_lines.append("")
            report_lines.append("Alignment sanity (examples):")
            for line in examples:
                report_lines.append(f"- {line}")

        report = "\n".join(report_lines)
        with open(self.config.report_path, "w", encoding=self.config.encoding_protocol) as f:
            f.write(report)

        self.logger.info(report)

if __name__ == "__main__":
    import os
    from src.misc.data_structures import FeatureReport, PolicyReport

    rr = ResultReporter(config=Config())
    fr = FeatureReport(r2_baseline=0.0, r2_with_llm=0.1, delta=0.1)
    pr = PolicyReport(
        mean_rewards_baseline=0.01,
        mean_rewards_linucb=0.02,
        pos_overall_base=0.50,
        pos_overall_linucb=0.60,
    )
    rr.generate_report(feature_report=fr, policy_report=pr, num_sessions=3, num_features=8, action_keys=['a', 'b'])
    assert os.path.exists(rr.config.report_path), "report.md was not written"
    print("[ResultReporter] OK: wrote", rr.config.report_path)

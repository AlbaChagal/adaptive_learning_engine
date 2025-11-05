import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from typing import List

from src.misc.config import Config
from src.misc.data_structures import FeatureReport, PolicyReport
from src.logging.logger import Logger


class LosoEvaluator:
    """
    Leave-One-Session-Out evaluator for regression tasks
    """
    def __init__(self, config: Config, is_debug: bool = False):
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                                     logging_level="debug" if is_debug else "info")

    def eval_regression(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform Leave-One-Session-Out evaluation using Ridge regression.
        :param X: Feature matrix
        :param y: Target vector
        :return: R^2 score
        """
        self.logger.info(f'eval_regression - Starting LOSO evaluation with {len(y)} '
                         f'samples and X of shape {X.shape}.')
        preds: np.ndarray = np.zeros_like(y)
        n: int = len(y)

        indices: List[int]
        model: Ridge
        for i in range(n):
            indices = [j for j in range(n) if j != i]
            model = Ridge(alpha=self.config.loso_ridge_alpha)
            model.fit(X[indices], y[indices])
            preds[i] = model.predict(X[i: i + 1])[0]
        score: float = float(r2_score(y, preds))
        self.logger.debug(f'eval_regression - Predictions: {preds}')
        self.logger.info(f'eval_regression - LOSO score: {score:.4f}')
        return score

    def eval_pair(self, X_base: np.ndarray, X_llm: np.ndarray, y: np.ndarray) -> FeatureReport:
        """
        Compare baseline features with LLM-augmented features using LOSO evaluation.
        :param X_base: Baseline feature matrix
        :param X_llm: LLM feature matrix
        :param y: Target vector
        :return: Dictionary with R^2 scores and their difference
        """
        self.logger.info(f'eval_pair - Evaluating X_base of shape {X_base.shape} '
                         f'and X_llm of shape {X_llm.shape} with {len(y)} samples.')

        if len(y) < 3:
            raise ValueError("Not enough data points for LOSO evaluation.")
        r2_base: float = self.eval_regression(X_base, y)
        r2_llm: float  = self.eval_regression(np.hstack([X_base, X_llm]), y)
        delta: float = r2_llm - r2_base
        self.logger.info(f'eval_pair - R2 baseline: {r2_base:.4f}, R2 with LLM: {r2_llm:.4f}, '
                         f'Delta: {delta:.4f}')
        return FeatureReport(r2_baseline=r2_base,
                             r2_with_llm=r2_llm,
                             delta=delta)

    @staticmethod
    def compare_policies(rewards_baseline: List[float],
                         rewards_linucb: List[float],
                         pos_overall_base: float,
                         pos_overall_lin: float) -> PolicyReport:
        """
        Compare two evaluation based on their rewards and overall position rates.
        :param rewards_baseline: List of rewards from the baseline policy
        :param rewards_linucb: List of rewards from the LinUCB policy
        :param pos_overall_base: Overall position rate for the baseline policy
        :param pos_overall_lin: Overall position rate for the LinUCB policy
        :return: PolicyReport containing mean rewards and overall position rates
        """
        return PolicyReport(
            mean_rewards_baseline=float(np.mean(rewards_baseline)) if rewards_baseline else 0.0,
            mean_rewards_linucb=float(np.mean(rewards_linucb)) if rewards_linucb else 0.0,
            pos_overall_base=pos_overall_base,
            pos_overall_linucb=pos_overall_lin,
        )

    @staticmethod
    def calc_pos_overall_rate(pos_overall: int, num_sessions: int) -> float:
        """
        Calculate the overall positive rate.
        :param pos_overall: The overall positive count
        :param num_sessions: The number of sessions
        :return: The overall positive rate
        """
        pos_overall_rate: float = pos_overall / max(1, (num_sessions - 1))
        return pos_overall_rate


if __name__ == "__main__":

    ev = LosoEvaluator(config=Config(), is_debug=True)
    # Tiny synthetic pair: 4 examples, 3 baseline feats, 3 llm feats
    Xb = np.array([[0.1,0.2,0.3],[0.3,0.1,0.2],[0.2,0.2,0.2],[0.4,0.1,0.1]], dtype=float)
    Xl = np.array([[0.1,0.2,0.3],[0.3,0.1,0.2],[0.2,0.2,0.2],[0.4,0.1,0.1]], dtype=float)
    y  = np.array([0.0, 0.1, -0.05, 0.2], dtype=float)
    fr = ev.eval_pair(X_base=Xb, X_llm=Xl, y=y)
    assert np.isfinite(fr.r2_baseline) and np.isfinite(fr.r2_with_llm), "R² must be finite"
    pr = ev.compare_policies(rewards_baseline=[0.0, 0.1], rewards_linucb=[0.2, 0.1], pos_overall_base=0.5, pos_overall_lin=0.6)
    assert pr.mean_rewards_linucb >= 0.0, "PolicyReport sanity"
    print("[LosoEvaluator] OK: R2_base=", fr.r2_baseline, "R2_llm=", fr.r2_with_llm)

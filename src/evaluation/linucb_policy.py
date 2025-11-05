import numpy as np
from typing import List, Dict, Tuple

from src.misc.config import Config
from src.logging.logger import Logger
from src.evaluation.policy import Policy


class LinUCB(Policy):
    """
    LinUCB policy for contextual multi-armed bandits
    1. Initialize A and b for each arm
    2. For each round:
       a. For each arm, compute UCB score
       b. Select arm with highest UCB score
       c. Observe reward and update A and b for the selected arm
    3. Repeat
    4. Parameters:
       - n_features: Number of features in context vector
       - arms: List of arm names
       - alpha: Exploration parameter
       - reg: Regularization parameter
    5. Methods:
       - select(x, last_actions, max_repeat): Select an arm based on context x
       - update(arm, x, reward): Update model parameters for the selected arm
    6. Usage:
       bandit = LinUCB(n_features=10, arms=['clarity', 'active_listening'], alpha=0.4, reg=1e-3)
       chosen_arm = bandit.select(x, last_actions)
       bandit.update(chosen_arm, x, reward)
    """
    def __init__(self,
                 config: Config,
                 n_features: int,
                 arms: List[str],
                 is_debug: bool = False):

        super().__init__(config=config, is_debug=is_debug)
        self.arms: List[str] = arms
        self.alpha: float = float(self.config.linucb_alpha)
        self.logger: Logger = Logger(self.__class__.__name__, logging_level="debug" if is_debug else "info")
        self.A: Dict[str, np.ndarray] = \
            {a: np.eye(n_features) * self.config.linubc_regularization_factor for a in arms}
        self.b: Dict[str, np.ndarray] = \
            {a: np.zeros((n_features, 1)) for a in arms}

    def compute(self, a: str, x: np.ndarray) -> float:
        """
        Compute UCB score for arm a given context x
        :param a: The arm name
        :param x: The context vector
        :return: The UCB score
        """
        self.logger.info(f'Computing UCB for arm: {a}, for vector x of shape: {x.shape}')
        A_inv: np.ndarray = np.linalg.inv(self.A[a])
        theta: np.ndarray = A_inv @ self.b[a]
        mu: float = float((theta.T @ x)[0, 0])
        sigma: float = float(np.sqrt(x.T @ A_inv @ x))

        final: float = mu + self.alpha * sigma

        self.logger.debug(f'_ucb - A_inv shape: {A_inv.shape}, '
                          f'theta shape: {theta.shape}, '
                          f'self.b[a] shape: {self.b[a].shape}, '
                          f'mu: {mu}, '
                          f'sigma: {sigma}')
        self.logger.info(f'_ucb - final score: {final}')

        return final

    def select(self, x: np.ndarray, last_actions: List[str], max_repeat: int = 3) -> str:
        """
        Select an arm based on context x and last actions
        :param x: The context vector
        :param last_actions: List of last selected actions
        :param max_repeat: Maximum allowed repeats of the same action
        :return: The selected arm name
        """
        self.logger.info(f'select - Selecting arm for vector x of shape: {x.shape}, '
                         f'last_actions: {last_actions}, '
                         f'max_repeat: {max_repeat}')
        scores: Dict[str, float] = {a: self.compute(a, x) for a in self.arms}
        chosen: str = max(scores, key=scores.get)
        # avoid > max_repeat same focus in a row
        if len(last_actions) >= max_repeat and all(la == last_actions[-1] for la in last_actions[-max_repeat:]):
            # pick the best alternative
            alternatives: List[Tuple[str, float]] = (
                sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
            for a, _ in alternatives:
                if a != last_actions[-1]:
                    chosen = a
                    break

        self.logger.debug(f'select - scores: {scores}')
        self.logger.info(f'select - chosen arm: {chosen} with score: {scores[chosen]}')

        return chosen

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        """
        Update model parameters for the selected arm
        :param arm: The selected arm name
        :param x: The context vector
        :param reward: The observed reward
        :return: None
        """
        self.logger.info(f'update - Updating arm: {arm} with reward: {reward} for vector x of shape: {x.shape}')
        x: np.ndarray = x.reshape((-1,1))
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

        self.logger.info(f'update - Updated A[{arm}] shape: {self.A[arm].shape}, '
                         f'Updated b[{arm}] shape: {self.b[arm].shape}')


if __name__ == "__main__":
    cfg = Config()
    arms = ["clarity", "active_listening", "call_to_action", "friendliness"]
    bandit = LinUCB(config=cfg, n_features=4, arms=arms, is_debug=True)

    x = np.array(
        object=[[1.0], [0.0], [0.0], [0.0]]
        , dtype=float
    )
    last = ["clarity", "clarity", "clarity"]
    choice = bandit.select(x=x, last_actions=last, max_repeat=3)
    assert choice != "clarity", "Safety rule failed: should avoid >3 repeats"
    bandit.update(arm=choice, x=x, reward=0.1)
    print("[LinUCB] OK: choice=", choice)

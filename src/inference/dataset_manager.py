import json
import os
import random
from typing import List, Any

from src.misc.config import Config
from src.misc.data_structures import Session
from src.logging.logger import Logger
from src.misc.types import DataSetType


class DatasetManager(object):
    """
    A class to manage dataset loading and synthetic dataset creation
    """

    def __init__(self, config: Config, is_debug: bool = False):
        """
        Initialize the dataset manager
        :param config: The configuration object
        :param is_debug: Whether to enable debug logging
        """
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                             logging_level="debug" if is_debug else "info")
        self.random_state: random.Random = random.Random(config.random_seed)
        self.round_accuracy: int = 3

    def read_json(self, path: str) -> Any:
        """
        Read a JSON file from the given path
        :param path: The path to the JSON file
        :return: The loaded JSON object
        """
        self.logger.debug(f'Reading JSON file from {path}')
        with open(path, "r", encoding=self.config.encoding_protocol) as f:
            return json.load(f)

    @staticmethod
    def _write_json(obj: Any, path: str, encoding: str) -> None:
        """
        Write a JSON object to the given path
        :param obj: The JSON object to write
        :param path: The path to the JSON file
        :param encoding: The encoding to use
        :return: None
        """
        with open(path, "w", encoding=encoding) as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def write_json(self, obj, path: str) -> None:
        """
        Write a JSON object to the given path
        :param obj: The JSON object to write
        :param path: The path to the JSON file
        :return: None
        """
        self.logger.debug(f'Writing JSON file to {path}')
        self._write_json(obj=obj, path=path, encoding=self.config.encoding_protocol)

    @staticmethod
    def _normalize(value: float) -> float:
        """
        Normalize a float value to [0,1]
        :param value: The float value
        """
        return max(0.0, min(1.0, value))

    def _round(self, val: float) -> float:
        """
        Round a float to the specified accuracy
        :param val: The float value
        """
        return round(val, self.round_accuracy)

    @staticmethod
    def _synthetic_text(i: int) -> str:
        """
        Generate synthetic text for a turn
        :param i: An integer to vary the text
        :return: A synthetic text string
        """
        raise NotImplementedError("Synthetic text generation not implemented. "
                                  "Should be implemented utilizing LLMs")

    def _generate_synthetic_dataset(self, path: str, n: int = 8) -> None:
        """
        Create a synthetic dataset and save it to the given path
        :param path: The path to save the synthetic dataset
        :param n: The number of sessions to create
        :return: None
        """
        raise NotImplementedError("Synthetic dataset generation not implemented. "
                                  "Should be implemented utilizing LLMs")
        speaker: str
        sessions: List[Session] = []
        overall: float = 0.45
        clarity: float = 0.45
        active_listening: float = 0.45
        call_to_action: float = 0.45
        friendliness: float = 0.55

        for sess_id in range(1, n + 1):
            # random walk with gentle drifts upward
            overall          += self.random_state.uniform(-0.03, b=0.05) * 100
            clarity          += self.random_state.uniform(-0.04, b=0.06)
            active_listening += self.random_state.uniform(-0.04, b=0.06)
            call_to_action   += self.random_state.uniform(-0.04, b=0.06)
            friendliness     += self.random_state.uniform(-0.03, b=0.03)

            # Normalize and round
            overall          = self._round(self._normalize(overall))
            clarity          = self._round(self._normalize(clarity))
            active_listening = self._round(self._normalize(active_listening))
            call_to_action   = self._round(self._normalize(call_to_action))
            friendliness     = self._round(self._normalize(friendliness))

            sessions.append(Session(
                session_id=sess_id,
                transcript=self._synthetic_text(sess_id),
                overall=overall,
                clarity=clarity,
                active_listening=active_listening,
                call_to_action=call_to_action,
                friendliness=friendliness,
            ))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.write_json(obj=sessions, path=path)

    def load_dataset(self,
                     path: str,
                     is_create_synthetic_dataset: bool = False) -> List[Session]:
        """
        Load the dataset from the given path,
        creating a synthetic dataset if specified
        :param path: The path to the dataset file
        :param is_create_synthetic_dataset: Whether to create a synthetic dataset
        :return: A list of Session objects
        """
        self.logger.info(f'load_dataset - Loading dataset from {path}, '
                         f'is_create_synthetic_dataset={is_create_synthetic_dataset}')
        if is_create_synthetic_dataset:
            if os.path.exists(path):
                raise FileExistsError(f"Synthetic dataset file already exists at {path}")
            else:
                self.logger.info(f'Creating synthetic dataset at {path}')
                self._generate_synthetic_dataset(path)
        else:
            assert os.path.exists(path), f"Dataset file not found at {path}"

        raw: DataSetType = self.read_json(path)
        sessions: List[Session] = []
        for raw_sess in raw:
            sessions.append(Session(
                session_id=raw_sess["session_id"],
                transcript=raw_sess["transcript"],
                overall=float(raw_sess["rubrics"]["overall"]),
                clarity=float(raw_sess["rubrics"]["clarity"]),
                active_listening=float(raw_sess["rubrics"]["active_listening"]),
                call_to_action=float(raw_sess["rubrics"]["call_to_action"]),
                friendliness=float(raw_sess["rubrics"]["friendliness"]),
            ))
        self.logger.info(f'Loaded {len(sessions)} sessions from dataset')
        return sessions

if __name__ == "__main__":
    from pathlib import Path
    import json

    dm = DatasetManager(config=Config(), is_debug=True)
    data_path = Path(dm.config.dataset_file_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        demo = [{
            "session_id": 0,
            "transcript": "Hi, maybe we can try tomorrow?",
            "rubrics": {"overall": 70, "clarity": 0.6, "active_listening": 0.5, "call_to_action": 0.4, "friendliness": 0.8}
        }]
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(demo, f)

    sessions = dm.read_json(dm.config.dataset_file_path)
    assert isinstance(sessions, list) and len(sessions) >= 1, "DatasetManager failed to load sessions"
    assert "transcript" in sessions[0] and "rubrics" in sessions[0], "Session schema missing keys"
    print("[DatasetManager] OK:", len(sessions), "sessions loaded")

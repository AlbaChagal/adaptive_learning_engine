from dataclasses import dataclass
from typing import List, Dict, Union, Any


@dataclass
class Session:
    """
    Represents a coaching session with its transcript and various ratings.
    """
    session_id: int
    transcript: str
    overall: float
    clarity: float
    active_listening: float
    call_to_action: float
    friendliness: float

@dataclass
class PolicyReport:
    """
    Represents a report comparing baseline and LinUCB policy performance.
    """
    mean_rewards_baseline: float
    mean_rewards_linucb: float
    pos_overall_base: float
    pos_overall_linucb: float

@dataclass
class FeatureReport:
    """
    Represents a report comparing R² values for baseline and LLM-enhanced models.
    """
    r2_baseline: float
    r2_with_llm: float
    delta: float

@dataclass
class CoachingCard:
    """
    Represents a coaching card with focus, rationale, exercises, scenario, and upgrades.
    """
    focus: str
    why: str
    exercises: List[str]
    scenario: Dict[str, Union[str, List[str]]]
    upgrades: List[str]

@dataclass
class CoachingLog:
    """
    Represents a log entry for a coaching session, including turn, focus, and associated coaching card
    """
    turn: int
    focus: str
    card: CoachingCard

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the CoachingLog instance to a dictionary.
        :return: A dictionary representation of the CoachingLog instance
        """
        return {
            "turn": self.turn,
            "focus": self.focus,
            "card": {
                "focus": self.card.focus,
                "why": self.card.why,
                "exercises": self.card.exercises,
                "scenario": self.card.scenario,
                "upgrades": self.card.upgrades,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CoachingLog':
        """
        Creates a CoachingLog instance from a dictionary.
        :param data: The dictionary containing coaching log data
        :return: A CoachingLog instance
        """
        card_data: Dict = data["card"]
        card: CoachingCard = CoachingCard(
            focus=card_data["focus"],
            why=card_data["why"],
            exercises=card_data["exercises"],
            scenario=card_data["scenario"],
            upgrades=card_data["upgrades"],
        )
        return cls(
            turn=data["turn"],
            focus=data["focus"],
            card=card
        )
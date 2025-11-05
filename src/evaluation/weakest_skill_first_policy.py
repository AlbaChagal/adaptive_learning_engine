from typing import Dict

from src.misc.config import Config
from src.misc.data_structures import Session
from src.logging.logger import Logger
from src.evaluation.policy import Policy
from src.misc.types import NumberType


class WeakestSkillFirst(Policy):
    """
    A policy that always selects the skill with the weakest performance
    for coaching focus.
    """

    def __init__(self, config: Config, is_debug: bool = False):
        super().__init__(config=config, is_debug=is_debug)
        self.logger = Logger(self.__class__.__name__,
                             logging_level="debug" if is_debug else "info")

    def select(self, sess: Session) -> str:
        """
        Select the weakest skill for coaching focus
        :param sess: The session to select the weakest skill from
        :return: The weakest skill as a string
        """
        skills: Dict[str, NumberType] = {
            "clarity": sess.clarity,
            "active_listening": sess.active_listening,
            "call_to_action": sess.call_to_action,
            "friendliness": sess.friendliness,
        }

        skill: str = min(skills, key=skills.get)
        self.logger.debug(f'Skill scores: {skills}')
        self.logger.info(f'select - Selecting weakest skill: {skill} for session_id: {sess.session_id}')
        return skill

    def update(self, *args, **kwargs):
        """
        No update needed for this policy
        """
        pass

    def compute(self, *args, **kwargs):
        """
        No compute needed for this policy
        """
        pass


if __name__ == "__main__":
    ws = WeakestSkillFirst(config=Config(), is_debug=True)
    sess = Session(clarity=0.6,
                   active_listening=0.5,
                   call_to_action=0.4,
                   friendliness=0.8,
                   session_id=0,
                   transcript='',
                   overall=100.)
    choice = ws.select(sess)

    if isinstance(choice, str):
        assert choice == "call_to_action", f"Expected 'call_to_action' as weakest, got {choice}"
        print("[WeakestSkillFirst] OK:", choice)
    else:
        print("[WeakestSkillFirst] OK (no-op baseline)")

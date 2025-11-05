import hashlib
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, \
    PreTrainedTokenizer, PreTrainedModel, Pipeline
import torch
from typing import Dict, Tuple, Optional

from src.misc.config import Config
from src.logging.logger import Logger


class LLMClient:
    """
    Deterministic stub. Replace this with real API calls later.
    """
    def __init__(self, config: Config, is_debug: bool = False):
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__,
                             logging_level="debug" if is_debug else "info")
        self.model_name: str = self.config.llm_model_name
        self.temperature: float = self.config.llm_temperature
        self.separator: str = self.config.llm_hash_seperator
        self.encoder_protocol: str = self.config.encoding_protocol
        self.device: int
        torch_dtype: torch.dtype
        self.device, torch_dtype = self._get_device_and_torch_dtype()

        self.logger.info(f"Loading local model: {self.model_name}")
        self.tokenizer: PreTrainedTokenizer = (
            AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True))
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == 0 else None
        )

        self.pipe: Pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.eos_token_id: str = self.tokenizer.eos_token_id

    @staticmethod
    def _get_device_and_torch_dtype() -> Tuple[int, torch.dtype]:
        """
        Get the device and torch dtype to use
        :return: A tuple of (device, torch_dtype)
        """
        use_cuda: bool = torch.cuda.is_available()
        use_mps: bool = torch.backends.mps.is_available()
        device: int = 0 if use_cuda else -1
        torch_dtype: torch.dtype = torch.float16 if (use_cuda or use_mps) else torch.float32
        return device, torch_dtype

    def post_process_output(self, output: str) -> float:
        """
        Post-process the output text to make sure it is a float in [0,1] and if not, try to force it
        :param output: The raw output text
        :return: The post-processed text
        """
        reg: re.Pattern = re.compile(r"([0-1](?:\.\d+)?)")
        match: Optional[re.Match] = reg.search(output)
        if match:
            val: float = float(match.group(1))
            return max(0.0, min(1.0, val))
        else:
            self.logger.warning(f"post_process_output - Could not parse float from output: {output}. "
                                f"Defaulting to 0.0")
            return 0.0

    def __call__(self, prompt: str, *args, **kwargs) -> float:
        """
        Generate text locally. Deterministic (no sampling).
        Returns the generated suffix (without echoing the prompt).
        :param prompt: The input prompt
        :return: The generated text suffix
        """
        self.logger.debug(f'call - Generating text for prompt (len={len(prompt)})')
        self.logger.info(f'call - Calling local LLM, prompt length={len(prompt)}')

        # Deterministic decoding
        torch.manual_seed(self.config.random_seed)
        outs = self.pipe(
            prompt,
            max_new_tokens=6,
            do_sample=False,
            temperature=0.0,  # ignored when do_sample=False, kept for clarity
            eos_token_id=self.eos_token_id
        )

        full: str = outs[0]["generated_text"]
        text: str = full[len(prompt):].strip() if full.startswith(prompt) else full
        self.logger.debug(f'call - Generated text (local): {text}')
        result: float = self.post_process_output(text)
        return result

    @staticmethod
    def _hash_to_str(text: str,
                     salt: str,
                     separator: str,
                     encoder_protocol: str = "utf-8") -> str:
        """
        Hash text with salt to a string
        :param text: An input text
        :param salt: The salt to use (for determinism)
        :param encoder_protocol: The encoding protocol to use
        :return: A string hash
        """
        salted: str = salt + separator + text
        encoded: bytes = salted.encode(encoder_protocol)
        hashed_str: str = hashlib.sha256(encoded).hexdigest()
        return hashed_str

    def _hash_to_01(self, text: str, salt: str) -> float:
        """
        Hash text with salt to a float in [0,1]
        :param text: An input text
        :param salt: The salt to use (for determinism)
        :return: A float in [0,1]
        """

        hashed_str: str = self._hash_to_str(text=text,
                                            salt=salt,
                                            separator=self.separator,
                                            encoder_protocol=self.encoder_protocol)
        val: float = int(hashed_str[:12], 16) / float(0xFFFFFFFFFFFF)
        return max(0.0, min(1.0, val))

    @staticmethod
    def get_map_features_to_salt() -> Dict[str, str]:
        """
        Map feature names to their salts
        """
        return {
            "objection_mirroring": "obj_mirr",
            "question_ratio_llm": "q_ratio",
            "cta_explicitness": "cta_exp",
            "empathy_markers": "empathy",
            "hedging_intensity": "hedge",
            "you_we_orientation": "youwe",
            "structure_cues": "struct",
            "pushiness_vs_collab": "pushcollab",
        }

    def score_features(self, transcript: str) -> Dict[str, float]:
        """
        Score the transcript on various features
        :param transcript: The conversation transcript
        :return: A dict of feature scores rounded to 3 decimal places
        """
        features: Dict[str, float] = \
            {k: self._hash_to_01(transcript, salt=v)
             for k, v in self.get_map_features_to_salt().items()}
        rounded_features: Dict[str, float] = \
            {k: round(float(v), 3) for k, v in features.items()}
        return rounded_features


if __name__ == "__main__":
    client = LLMClient(config=Config(), is_debug=True)

    assert hasattr(client, "config"), "LLMClient missing config"
    print("[LLMClient] OK: initialized (API calls not executed in s"
          "elf-test test within end-to-end test via main.py)")

import re
import json
import os
import hashlib
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Callable, Tuple, Any

from src.logging.logger import Logger
from src.misc.types import ContextRowType, NumberType


class FeaturesExtractor:
    """
    Extract persuasion-related features from a session transcript.

    Accepts transcript as:
      - List[dict] where each dict has 'speaker' and 'text'
      - Or a single string (will be split into turns using simple heuristics)

    Features produced (all in [0,1]):
      - question_ratio: questions / total user turns (higher means more questions to user)
      - empathy_markers: fraction of agent turns containing empathy phrases
      - hedging_intensity: normalized density of hedging words in agent utterances
      - you_we_orientation: proportion of "you" vs ("you"+"we") usage (higher => more customer-centric)
      - structure_cues: presence/strength of structural cues (numbered steps, "first", "to recap", summaries)
      - cta_explicitness: presence and strength of explicit call-to-action + timeframe
      - objection_mirroring: degree agent echoes user concerns before rebuttal (overlap-based)
      - pushiness_vs_collaborative: inverse of pushiness (higher => more collaborative)

    Hybrid approach:
      - Lightweight deterministic rules compute base scores.
      - Optional LLM callable (deterministic prompt + seed) can be used to refine a score per feature.
      - All LLM-call results are cached on disk to keep runs deterministic and fast.
    """

    HEDGES: List[str] = ["maybe", "might", "could", "sort of", "kind of", "perhaps", "possibly", "seem"]
    EMPATHY_PHRASES: List[str] = ["i understand", "i'm sorry", "i am sorry", "i can imagine",
                                  "that sounds", "i see", "thank you for", "thanks for", "that must be"]
    CTA_VERBS: List[str] = ["sign up", "schedule", "call", "book", "register", "start",
                            "subscribe", "download", "visit", "reply", "confirm"]
    PUSHY_PHRASES: List[str] = ["must", "only this week", "limited time", "don't miss",
                                "act now", "no time", "final offer", "guaranteed", "urgent"]
    STRUCTURE_CUES: List[str] = ["first", "second", "third", "step", "steps",
                                 "to recap", "in summary", "summary", "next", "finally"]

    def __init__(self,
                 cache_path,
                 seed: int,
                 is_debug: bool = False,
                 llm_callable: Callable = None):
        """
        :param cache_path: where to store LLM call cache (json)
        :param seed: deterministic seed for pseudo-LLM
        :param use_real_llm: if True, will call llm_callable for feature refinements
        :param llm_callable: function(prompt:str) -> float in [0,1]
        :param is_debug: if True, enable debug logging (currently unused)
        :return: None
        """
        self.logger: Logger = Logger(self.__class__.__name__, logging_level="debug" if is_debug else "info")
        self.cache_path: str = cache_path
        self.seed: int = seed
        self.is_use_real_llm: bool = llm_callable is not None
        self.llm_callable: Callable = llm_callable
        self._ensure_cache_file()

    def __len__(self):
        return len(self.get_feature_names())

    def _ensure_cache_file(self) -> None:
        """
        Ensure the cache file exists; if not, create an empty one.
        """
        self.logger.info(f'_ensure_cache_file - '
                         f'Ensuring cache file at: {self.cache_path}')
        if not os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "w") as f:
                    json.dump({}, f)
                self.logger.info(f'_ensure_cache_file - '
                                 f'Created new cache file at: {self.cache_path}')
            except OSError:
                # best-effort: if cannot write, continue without persistent cache
                self.logger.warning('_ensure_cache_file - Could not create cache file; '
                                 'continuing without persistent cache.')
                pass


    def _cache_get(self, key: str) -> Any:
        """
        Retrieve a value from the cache by key.
        :param key: The cache key
        :return: The cached value or None if not found
        """
        self.logger.info(f'_cache_get - Retrieving key: {key} '
                         f'from cache at: {self.cache_path}')
        try:
            with open(self.cache_path, "r") as f:
                data: Any = json.load(f)
            self.logger.debug(f'_cache_get - Cache data: {data}')
            self.logger.info(f'_cache_get - Cache data loaded successfully.')
            return data.get(key)
        except Exception:
            self.logger.warning(f'_cache_get - Error reading cache file')
            return None

    def _cache_set(self, key: str, value: Any):
        """
        Store a value in the cache by key.
        :param key: The cache key
        :param value: The value to cache
        :return: None
        """
        self.logger.info(f'_cache_set - Setting key: {key}, value type: {type(value)}')
        try:
            # read-modify-write to avoid stomping other processes
            data: Dict = {}
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = {}
            data[key] = value
            with open(self.cache_path, "w") as f:
                json.dump(data, f)
            self.logger.info(f'_cache_set - Cache updated successfully.')
        except Exception:
            self.logger.warning(f'_cache_set - Error writing to cache file')
            pass

    def _hash(self, text: str) -> str:
        """
        Hash text with seed to produce a deterministic hash string.
        :param text: The text to hash
        :return: The resulting hash string
        """
        self.logger.debug(f'_hash - Hashing text with seed: {self.seed}, text length: {len(text)}, text: {text}')
        h: hashlib._Hash = hashlib.sha256()
        h.update(f"{self.seed}:{text}".encode("utf-8"))
        hashed_text: str = h.hexdigest()
        self.logger.debug(f'_hash - Hash digest: {hashed_text}')
        return hashed_text

    def _pseudo_llm_score(self, prompt: str) -> float:
        """
        Deterministic pseudo-LLM: map prompt -> float in [0,1] via hash.
        Used if real LLM isn't configured. Cache results.
        :param prompt: The prompt string
        :return: The pseudo-LLM score
        """
        self.logger.info(f'_pseudo_llm_score - Generating pseudo-LLM score for prompt length: {len(prompt)}')
        key: str = "pseudo:" + self._hash(prompt)
        cached: Any = self._cache_get(key)
        if cached is not None:
            self.logger.info(f'_pseudo_llm_score - Found cached value: {cached}, for key: {key}')
            return float(cached)
        digest: str = self._hash(prompt + ":pseudo")
        # take first 8 hex chars to int, normalize to [0,1]
        v: float = int(digest[:8], 16) / float(0xFFFFFFFF)
        self._cache_set(key, v)
        self.logger.info(f'_pseudo_llm_score - Computed pseudo-LLM score: {v}, for key: {key}')
        return v

    def _call_llm_or_pseudo(self, prompt: str) -> float:
        """
        Deterministic call: if a real llm callable is configured, use it and cache results.
        Otherwise use the pseudo deterministic function.
        The llm_callable must be deterministic for the same prompt and seed.
        :param prompt: The prompt string
        :return: The resulting score in [0,1]
        """
        self.logger.info(f'_call_llm_or_pseudo - Calling LLM or pseudo for prompt length: {len(prompt)}')
        self.logger.debug(f'_call_llm_or_pseudo - Prompt: {prompt}')
        key: str = ("llm:" if self.is_use_real_llm else "pseudo:") + self._hash(prompt)
        cached: Any = self._cache_get(key)
        if cached is not None:
            self.logger.info(f'_pseudo_llm_score - Found cached value: {cached}, for key: {key}')
            return float(cached)
        if self.is_use_real_llm and self.llm_callable:
            self.logger.info(f'_call_llm_or_pseudo - Using real LLM callable for key: {key}')
            # wrap call (caller ensures determinism)
            raw_val: float = float(self.llm_callable(prompt))
            val: float = max(0.0, min(1.0, raw_val))
            self.logger.debug(f'_call_llm_or_pseudo - raw LLM returned value: {raw_val}')
            self.logger.debug(f'_call_llm_or_pseudo - capped LLM returned value: {val}')
        else:
            self.logger.info(f'_call_llm_or_pseudo - Using pseudo LLM for key: {key}')
            val: float = self._pseudo_llm_score(prompt)
        self._cache_set(key, float(val))
        return float(val)

    # --------------------
    # Transcript helpers
    # --------------------
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, strip, collapse whitespace.
        :param text: The text to normalize
        :return: The normalized text
        """
        self.logger.debug(f'_normalize_text - Normalizing text of length: {len(text)}, text: \n{text}')
        normalized: str = re.sub(r"\s+", " ", text.strip().lower())
        self.logger.debug(f'_normalize_text - Normalized text: \n{normalized}')
        return normalized

    def _split_transcript(self, transcript: Union[List[Dict], str]) -> List[Dict]:
        """
        Return list of turns: {'speaker': str, 'text': str}
        If input is list, try to use it as-is; if string, split on newlines and heuristically assign speakers.
        :param transcript: The transcript input
        :return: List of turns with 'speaker' and 'text'
        """
        self.logger.info(f'_split_transcript - Splitting transcript of type: {type(transcript)}')
        self.logger.debug(f'_split_transcript - Splitting transcript: \n{transcript}')
        turns: List[Dict[str, str]]
        speaker: str
        if isinstance(transcript, list):
            turns = []
            for turn in transcript:
                if isinstance(turn, dict) and 'text' in turn:
                    speaker = turn.get('speaker', '').lower() if 'speaker' in turn else ''
                    turns.append({'speaker': speaker, 'text': self._normalize_text(turn['text'])})
            self.logger.debug(f'_split_transcript - Parsed turns: {turns}')
            return turns
        else:
            text: str = str(transcript)
            parts: List[str] = [p.strip() for p in text.splitlines() if p.strip()]
            turns = []

            m: re.Match
            txt: str
            for p in parts:
                m = re.match(r"^(user|agent|customer|client|rep|assistant)[:\-]\s*(.*)$", p, re.I)
                if m:
                    speaker = str(m.group(1).lower())
                    txt = m.group(2)
                else:
                    # alternate assignment: odd lines user, even agent
                    speaker = 'user' if len(turns) % 2 == 0 else 'agent'
                    txt = p
                turns.append({'speaker': speaker, 'text': self._normalize_text(txt)})
            self.logger.debug(f'_split_transcript - Parsed turns: {turns}')
            return turns

    # --------------------
    # Feature computations (rule-based cores)
    # --------------------
    def _question_ratio(self, turns: List[Dict]) -> float:
        """
        questions / total user turns (0..1). If no user turns, return 0.
        :param turns: List of turns
        :return: question ratio
        """
        self.logger.debug(f'_question_ratio - Calculating question ratio for {len(turns)} turns')
        user_turns: List[str] = [t for t in turns if t['speaker'] in ('user', 'customer', 'client')]
        if not user_turns:
            self.logger.warning(f'_question_ratio - No user turns found, returning 0.0')
            return 0.0

        pattern: str = r"\bwho\b|\bwhat\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\bwhich\b\b"
        q_count: int = sum(1 for t in user_turns if '?' in t['text'] or re.search(pattern, t['text']))
        ratio: float = q_count / len(user_turns)
        self.logger.debug(f'_question_ratio - Found {q_count} questions in {len(user_turns)} user turns, ratio: {ratio}')
        return ratio

    def _empathy_markers(self, turns: List[Dict]) -> float:
        """
        Fraction of agent turns that include empathy phrases.
        :param turns: List of turns
        :return: empathy markers score
        """
        self.logger.debug(f'_empathy_markers - Calculating empathy markers for {len(turns)} turns')
        agent_turns: List[str] = [t for t in turns if t['speaker'] not in ('user', 'customer', 'client')]
        if not agent_turns:
            self.logger.warning(f'_empathy_markers - No agent turns found, returning 0.0')
            return 0.0
        matches: int = 0
        txt: str
        for t in agent_turns:
            txt = t['text']
            for p in self.EMPATHY_PHRASES:
                if p in txt:
                    matches += 1
                    break

        empathy_score: float = matches / len(agent_turns)
        self.logger.debug(f'_empathy_markers - Found {matches} empathy matches in '
                          f'{len(agent_turns)} agent turns, score: {empathy_score}')
        return empathy_score

    def _hedging_intensity(self, turns: List[Dict]) -> float:
        """
        Normalized hedging density in agent text: hedging word occurrences / total agent words, clipped to [0,1].
        :param turns: List of turns
        :return: hedging intensity score
        """
        self.logger.debug(f'_hedging_intensity - Calculating hedging intensity for {len(turns)} turns')
        agent_text: str = " ".join([t['text'] for t in turns if t['speaker'] not in ('user', 'customer', 'client')])
        if not agent_text.strip():
            self.logger.warning(f'_hedging_intensity - No agent text found, returning 0.0')
            return 0.0
        words: List[str] = re.findall(r"\w+'?\w*|\w+", agent_text)
        total: int = max(1, len(words))
        hedges: int = 0
        for h in self.HEDGES:
            # count occurrences (simple)
            hedges += len(re.findall(r"\b" + re.escape(h) + r"\b", agent_text))

        score: float = min(1.0, hedges / total)
        self.logger.debug(f'_hedging_intensity - Found {hedges} hedges in {total} words, score: {score}')
        return score

    def _you_we_orientation(self, turns: List[Dict]) -> float:
        """
        Score = count('you') / (count('you') + count('we') + 1e-6)
        Higher => more customer-centric. If neither appears, return 0.5 (neutral).
        :param turns: List of turns
        :return: you/we orientation score
        """
        self.logger.debug(f'_you_we_orientation - Calculating you/we orientation for {len(turns)} turns')
        whole: str = " ".join([t['text'] for t in turns])
        you: int = len(re.findall(r"\byou\b", whole))
        we: int = len(re.findall(r"\bwe\b", whole))
        if you + we == 0:
            self.logger.warning(f'_you_we_orientation - Neither "you" nor "we" found, returning 0.5')
            return 0.5
        orientation: float = you / (you + we)
        self.logger.debug(f'_you_we_orientation - Found {you} "you" and {we} "we", score calculation, score: {orientation}')
        return orientation

    def _structure_cues(self, turns: List[Dict]) -> float:
        """
        Presence of structure cues: number of distinct cues found divided by total possible cues.
        Encourages explicit structure (numbered steps, summaries).
        :param turns: List of turns
        :return: structure cues score
        """
        self.logger.debug(f'_structure_cues - Calculating structure cues for {len(turns)} turns')
        agent_text: str = " ".join([t['text'] for t in turns if t['speaker'] not in ('user', 'customer', 'client')])
        found: int = 0
        for cue in self.STRUCTURE_CUES:
            if re.search(r"\b" + re.escape(cue) + r"\b", agent_text):
                found += 1
        score: float = min(1.0, found / max(1, len(self.STRUCTURE_CUES)))
        self.logger.debug(f'_structure_cues - Found {found} structure cues, score: {score}')
        return score

    def _cta_explicitness(self, turns: List[Dict]) -> float:
        """
        Detect explicit CTAs: presence of a CTA verb and indication of timeframe or explicit ask.
        Score between 0 and 1: 0 none, 1 explicit and timeframe.
        :param turns: List of turns
        :return: cta explicitness score
        """
        self.logger.debug(f'_cta_explicitness - Calculating CTA explicitness for {len(turns)} turns')
        agent_text: str = " ".join([t['text'] for t in turns if t['speaker'] not in ('user', 'customer', 'client')])
        has_cta: bool = any(re.search(r"\b" + re.escape(verb) + r"\b", agent_text) for verb in self.CTA_VERBS)
        has_timeframe: bool = bool(re.search(r"\btoday\b|\bby\b|\bwithin\b|\bthis week\b|\b24 hours\b|\btomorrow\b|\bnow\b", agent_text))
        has_please: bool = bool(re.search(r"\bplease\b|\bkindly\b", agent_text))
        score: float = 0.0
        if has_cta:
            score = 0.5
            if has_timeframe:
                score = 1.0
            elif has_please:
                score = 0.75
        self.logger.debug(f'_cta_explicitness - has_cta: {has_cta}, has_timeframe: {has_timeframe}, '
                          f'has_please: {has_please}, score: {score}')
        return float(score)

    def _objection_mirroring(self, turns: List[Dict]) -> float:
        """
        Measure whether the agent echoes user's concerns before rebutting:
        For each user turn that contains negative/objection cues, check if the next agent turn shares bigram overlap.
        Score = fraction of user objections that were mirrored (0..1).
        :param turns: List of turns
        :return: objection mirroring score
        """
        self.logger.debug(f'_objection_mirroring - Calculating objection mirroring for {len(turns)} turns')
        objection_cues: List[str] = ['not', "don't", "can't", 'no', 'rather', 'rather not',
                                     'problem', 'concern', 'issue', 'too expensive', 'too costly', 'hate']
        user_turns: List[Tuple[int, Dict]] = \
            [(i, t) for i, t in enumerate(turns) if t['speaker'] in ('user', 'customer', 'client')]
        if not user_turns:
            self.logger.warning(f'_objection_mirroring - No user turns found, returning 0.0')
            return 0.0
        mirrored: int = 0
        total: int = 0
        txt: str
        b1: set
        b2: set
        for idx, ut in user_turns:
            txt = ut['text']
            if any(cue in txt for cue in objection_cues):
                total += 1
                # look for next agent turn after this user turn
                next_agent = None
                for j in range(idx + 1, len(turns)):
                    if turns[j]['speaker'] not in ('user', 'customer', 'client'):
                        next_agent = turns[j]['text']
                        break
                if not next_agent:
                    continue
                # compute overlap of bigrams
                def bigrams(s):
                    toks = re.findall(r"\w+", s)
                    return {" ".join(toks[i:i+2]) for i in range(max(0, len(toks)-1))}
                b1 = bigrams(txt)
                b2 = bigrams(next_agent)
                if not b1:
                    continue
                overlap = len(b1 & b2) / len(b1)
                if overlap >= 0.2:  # threshold for mirroring
                    mirrored += 1
        score: float = mirrored / total if total > 0 else 0.0
        self.logger.debug(f'_objection_mirroring - Mirrored {mirrored} out of {total} objections, score: {score}')
        return score

    def _pushiness_vs_collaborative(self, turns: List[Dict]) -> float:
        """
        Lower pushiness -> higher score. Detect pushy phrases; compute 1 - normalized_pushiness.
        If no agent text, return 0.5 neutral.
        :param turns: List of turns
        :return: pushiness vs collaborative score
        """
        self.logger.debug(f'_pushiness_vs_collaborative - '
                          f'Calculating pushiness vs collaborative for {len(turns)} turns')
        agent_text: str = " ".join([t['text'] for t in turns if t['speaker'] not in ('user', 'customer', 'client')])
        if not agent_text.strip():
            self.logger.warning(f'_pushiness_vs_collaborative - No agent text found, returning 0.5')
            return 0.5
        push_count: int = 0
        for p in self.PUSHY_PHRASES:
            push_count += len(re.findall(r"\b" + re.escape(p) + r"\b", agent_text))
        # normalize by agent words
        words: List[str] = re.findall(r"\w+'?\w*|\w+", agent_text)
        total: int = max(1, len(words))
        normalized_push: float = min(1.0, push_count / (math.sqrt(total)))
        score: float = 1.0 - normalized_push
        self.logger.debug(f'_pushiness_vs_collaborative - Found {push_count} pushy phrases in {total} words, '
                          f'normalized push: {normalized_push}, score: {score}')
        return score

    def get_baseline_feature_to_calc_func_dict(self) -> Dict[str, Callable]:
        """
        Get mapping of feature names to their calculation functions.
        :return: feature_name dict -> function
        """
        return {
            "question_ratio": self._question_ratio,
            "empathy_markers": self._empathy_markers,
            "hedging_intensity": self._hedging_intensity,
            "you_we_orientation": self._you_we_orientation,
            "structure_cues": self._structure_cues,
            "cta_explicitness": self._cta_explicitness,
            "objection_mirroring": self._objection_mirroring,
            "pushiness_vs_collaborative": self._pushiness_vs_collaborative
        }

    # --------------------
    # Public API
    # --------------------
    def extract_features(self, transcript: Union[List[Dict], str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute all features and optionally refine them via LLM prompts+cache.
        Returns dict feature_name -> float.
        :param transcript: The transcript input
        :return: Tuple of (baseline_features, llm_refined_features)
        """
        self.logger.info(f'extract_features - Extracting features from transcript of type: {type(transcript)}')
        self.logger.debug(f'extract_features - Transcript content: \n{transcript}')
        turns: List[Dict] = self._split_transcript(transcript)

        # base rule-derived features
        baseline_feature_to_calc_func_dict: Dict[str, Callable] = \
            self.get_baseline_feature_to_calc_func_dict()

        baseline_features: Dict[str, float] = \
            {key: float(value(turns)) for key, value in baseline_feature_to_calc_func_dict.items()}
        self.logger.debug(f'extract_features - Baseline features: {baseline_features}')

        # Optionally refine each feature using an LLM-style deterministic prompt + cache.
        # We build a short few-shot style prompt that asks for a 0..1 score for the given aspect.
        llm_features: Dict[str, float] = {}
        prompt: str
        llm_score: float
        for name, base_score in list(baseline_features.items()):
            prompt = (
                f"Deterministically score the {name} of this conversation between 0 and 1 (higher is better). "
                f"Base-rule score: {base_score:.3f}. Conversation:\n"
                + "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
                + f"\nReturn only a single float between 0 and 1, do not return anything except for that. Seed:{self.seed}"
            )
            # ask the llm or pseudo-llm for a refinement; then combine with base deterministically
            llm_score = self._call_llm_or_pseudo(prompt)
            llm_features[name] = llm_score
        self.logger.debug(f'extract_features - LLM-refined features: {llm_features}')
        return baseline_features, llm_features

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        :return: List of feature names
        """
        return list(self.get_baseline_feature_to_calc_func_dict().keys())

    @staticmethod
    def to_csv(rows: List[Dict], path: str) -> None:
        """
        Export features to CSV
        :param rows: A list of feature rows (dictionaries)
        :param path: The path to save the CSV
        :return: None
        """
        df: pd.DataFrame = pd.DataFrame(rows)
        df.to_csv(path, index=False)


    @staticmethod
    def create_context_vector(r: ContextRowType) -> np.ndarray:
        base_vec: List[NumberType] = [*(r["baseline_features"].values())]
        llm_vec: List[NumberType] = [*(r["llm_features"].values())]
        return np.array(base_vec + llm_vec + [r["delta_overall"], r["delta_skill_avg"]], dtype=float).reshape(-1, 1)


if __name__ == "__main__":
    from src.misc.config import Config

    is_debug_main = False
    config = Config()
    fx = FeaturesExtractor(
        cache_path=config.feature_cache_path,
        is_debug=is_debug_main,
        seed=config.random_seed,
        llm_callable=lambda *a, **k: 0.5,  # mock LLM: fixed score
    )
    row = {
        "transcript": "I understand. Maybe we could set a time? Tomorrow 10?",
        "rubrics": {"overall": 72, "clarity": 0.68, "active_listening": 0.59, "call_to_action": 0.71, "friendliness": 0.82},
        "baseline_features": {}
    }
    baseline_features1, llm_features1 = fx.extract_features([row])
    baseline_features2, llm_features2 = fx.extract_features([row])
    assert baseline_features1 == baseline_features2, "Baseline features must be deterministic with fixed seed + cache"
    assert llm_features1 == llm_features2, "LLM features must be deterministic with fixed seed + cache"
    assert len(llm_features1) >= 6, "Expected at least 6 LLM features"
    for k, v in llm_features1.items():
        assert 0.0 <= float(v) <= 1.0, f"Feature {k} out of [0,1]: {v}"
    print("[FeaturesExtractor] OK:", sorted(llm_features1.keys()))

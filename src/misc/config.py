import os

class Config:
    """
    A data structure class to hold all configuration related data
    """

    # Development constants
    is_debug_main: bool = False

    # Randomness
    random_seed: int = 42

    # LLM Constants
    llm_hash_seperator: str = "::"
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_temperature: float = 0.0

    # Paths & Directories
    artifacts_dir: str = "artifacts"
    coaching_log_filename: str = "coaching_log.json"
    dataset_file_path: str = "data/dataset.json"
    feature_cache_path: str = os.path.join(artifacts_dir, "features_cache.json")
    features_csv_path: str = os.path.join(artifacts_dir, "features.csv")
    report_path: str = os.path.join(artifacts_dir, "report.md")
    coaching_log_path: str = os.path.join(artifacts_dir, coaching_log_filename)
    coaching_next_filename: str = "coaching_next.json"
    coaching_next_path: str = os.path.join(artifacts_dir, coaching_next_filename)

    # Evaluation Constants
    focus_factor = 0.6
    overall_factor = 0.4
    baseline_feature_factor = 0.4
    llm_feature_factor = 0.6

    # LinUCB Constants
    linucb_alpha: float = 0.4
    linubc_regularization_factor: float = 1e-3
    linucb_max_focus_repeat: int = 3

    # LOSO Constants
    loso_ridge_alpha: float = 1.0

    # General Constants
    default_timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    encoding_protocol: str = "utf-8"


if __name__ == "__main__":
    c = Config()
    # Sanity-check - key paths exist or can be created
    os.makedirs(c.artifacts_dir, exist_ok=True)
    for p in [c.feature_cache_path, c.report_path, c.coaching_log_path, getattr(c, "coaching_next_path", c.report_path)]:
        assert isinstance(p, str) and len(p) > 0
    print("[Config] OK")

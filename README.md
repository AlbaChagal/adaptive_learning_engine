# Adaptive Learning Engine

This repository implements **An Adaptive Learning Engine** from the Convergent-AI Engineer Home Assignment.

The system builds an **end-to-end adaptive coaching pipeline** for persuasion skill training.  
It extracts LLM-based conversational features, learns an adaptive policy to select the next coaching focus and generates customized feedback and scenario stubs for the next practice session.

---

## 🧩 Project Overview

The implementation centers around the `AdaptiveLearningEngine` class (`src/main.py`), which coordinates all core modules:

| Component | Role |
|------------|------|
| **DatasetManager** | Loads and manages JSON persuasion sessions. |
| **FeaturesExtractor** | Computes numeric and LLM-based contextual features (cached, deterministic). |
| **LLMClient** | Interfaces with the chosen language model (e.g., GPT-4-turbo) for scoring and generation. |
| **LinUCB** | Adaptive contextual-bandit policy for focus selection. |
| **WeakestSkillFirst** | Baseline heuristic policy for comparison. |
| **CoachingCardGenerator** | Generates the next-step coaching card, micro-exercises, and scenario stubs. |
| **LosoEvaluator** | Performs leave-one-step-out evaluation for feature usefulness and model quality. |
| **ResultReporter** | Logs metrics and saves artifacts (features.csv, coaching_next.json, report.md). |

---

## 📁 Project Structure

```
adaptive-learning-engine/
├── src/
│   ├── main.py                       # Entry point: orchestrates full pipeline
│   ├── inference/
│   │   ├── dataset_manager.py        # Loads and parses persuasion sessions
│   │   ├── features.py               # Extracts LLM-based & numeric features
│   │   ├── llm_client.py             # Handles API calls and caching
│   │   ├── coaching_card_generator.py# Generates personalized coaching cards
│   ├── evaluation/
│   │   ├── linucb_policy.py          # Contextual bandit implementation
│   │   ├── weakest_skill_first_policy.py # Baseline policy
│   │   ├── evaluator.py              # LOSO evaluation and metrics
│   ├── logging/
│   │   └── result_reporter.py        # Summarizes outputs and writes reports
│   ├── misc/
│   │   ├── config.py                 # Centralized config & environment variables
│   │   ├── types.py                  # Shared type definitions
│   │   └── data_structures.py        # Dataclasses for reports and logs
├── artifacts/
│   ├── features.csv                  # Extracted baseline + LLM features
│   ├── coaching_next.json            # Focus decisions + coaching text
│   ├── report.md                     # Evaluation results
│   └── features_cache.json           # Cached LLM calls for reproducibility
├── data/
│   └── sessions.json                 # Input dataset (19 persuasion sessions)
├── requirements.txt
├── env.example
└── README.md
```

---

## Installation & Setup

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python -m src.main
```

The system will:
1. Load the dataset
2. Extract features (baseline + LLM)
3. Run adaptive policy (LinUCB)
4. Generate coaching cards and scenario stubs
5. Evaluate model performance and write results to `artifacts/`

NOTICE! The first run will take a longer time since the 
offline-LLM needs to be downloaded and the embeddings are not yet saved in cache

---

## 🧠 Key Features

- **LLM-driven analytics**: Extracts >6 contextual persuasion features such as empathy markers, CTA explicitness, and hedging intensity.
- **Adaptive bandit policy**: Chooses next skill focus using contextual LinUCB.
- **Coaching generation**: Produces 120–180 word feedback cards with exercises and progressive scenario stubs.
- **Deterministic runs**: Cached LLM calls and seeded randomness ensure reproducibility.
- **Lightweight evaluation**: LOSO cross-validation on overall delta; ablation of LLM vs. numeric-only features.

---

## 📊 Example Output

```
=== Coaching Policy Evaluation Report ===

Sessions: 19 | Steps: 18
Features: baseline=8 LLM=8
Policy = LinUCB (alpha=0.4) | Reward = 0.6*delta(focus)+0.4*delta(overall)

Feature ablation (LOSO):
- R2 baseline: -0.231
- R2 with LLM: -0.149
- R2 delta: 0.082

Policy comparison:
- Weakest-skill-first: mean reward=0.0088, overall_delta=50.00%
- LinUCB (+LLM feats): mean reward=0.0061, overall_delta=50.00%

Alignment sanity (examples):
- t=0: focus=clarity | weakest=active_listening (0.59) -> mismatch
- t=1: focus=active_listening | weakest=active_listening (0.63) -> match
- t=2: focus=call_to_action | weakest=active_listening (0.68) -> mismatch

```

---

## 🧩 Design Highlights

- Modular architecture enables independent debugging and testing of each component.
- Extensive use of **dataclasses** for structured reporting.
- Central **Config** object controls all paths, parameters, and debug flags.
- Feature and policy modules easily swappable for experimentation.

---

## 📦 Artifacts Generated

| File                            | Description                            |
|---------------------------------|----------------------------------------|
| `artifacts/features.csv`        | Tabular features per session           |
| `artifacts/coaching_next.json`  | Focus decisions and generated coaching |
| `artifacts/report.md`           | Evaluation summary                     |
| `artifacts/features_cache.json` | Cached LLM feature calls               |

---

## Self-tests (no setup required)
Each major class includes a quick sanity check under an if __name__ == "__main__": block.
These tests verify deterministic behavior, schema integrity, and correct artifact generation without any external dependencies or API calls.
Example:
```bash
python -m src.inference.features
python -m src.evaluation.linucb_policy
python -m src.inference.coaching_card_generator
```

Each command runs a local mini-test (with mock or cached data) and prints [OK] upon success.


| Module                | Test Coverage                                               |
|-----------------------|-------------------------------------------------------------|
| DatasetManager        | Validates dataset loading and schema                        |
| FeaturesExtractor     | Checks deterministic feature extraction and value ranges    |
| LLMClient             | Verifies initialization (no real API calls)                 |
| LinUCB                | Tests adaptive policy logic and safety rule                 |
| WeakestSkillFirst     | Ensures correct weakest-rubric selection                    |
| CoachingCardGenerator | Confirms coaching card structure (why, exercises, scenario) |
| LosoEvaluator         | Validates LOSO R^2 computation and policy comparison        |
| ResultReporter        | Checks report creation and alignment section                |
| Config                | Confirms valid paths and artifact directories               |

---

**Author:** Shachar Heyman  
**Date:** November 2025  
**Contact:** shahar@theheymans.com  
**GitHub:** [github.com/AlbaChagal](https://github.com/AlbaChagal)
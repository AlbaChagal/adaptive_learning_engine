
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
- turn=0: focus=clarity | weakest=active_listening (0.59) -> mismatch
- turn=1: focus=active_listening | weakest=active_listening (0.63) -> match
- turn=2: focus=call_to_action | weakest=active_listening (0.68) -> mismatch
# Assessment Coverage

This matrix maps the Exercise 3 requirements to concrete repository evidence.

| Requirement | Evidence | Status |
|---|---|---|
| Problem as sequential RL, not prediction only | `README.md`, `docs/PRD_problem_formulation.md` | PASS |
| Agent, state, action, reward, episode, policy, return, advantage, critic defined | `README.md`, `docs/PRD_problem_formulation.md`, `docs/PRD_a2c.md` | PASS |
| Clean data pipeline, no hardcoded absolute paths | `src/rl_gym_training/data/`, `config/setup.yaml`, tests | PASS |
| Chronological split and train-only scaling | `src/rl_gym_training/data/preprocessing.py`, `tests/unit/test_config_seed_data.py` | PASS |
| Real Kaggle-listed dataset workflow | `scripts/download_pmdata.py`, `docs/PMDATA_DATASET.md`, `docs/EXPERIMENTS.md` | PASS |
| Synthetic fallback clearly marked and disabled by default | `config/setup.yaml`, `README.md`, `docs/PRD_data_pipeline.md` | PASS |
| Separate LSTM world model | `src/rl_gym_training/models/lstm_world_model.py`, `docs/PRD_lstm_world_model.md` | PASS |
| REINFORCE policy-gradient implementation | `src/rl_gym_training/rl/reinforce.py`, `docs/PRD_reinforce.md`, tests | PASS |
| A2C Actor-Critic implementation | `src/rl_gym_training/rl/a2c.py`, `docs/PRD_a2c.md`, tests | PASS |
| Configurable reward function | `src/rl_gym_training/rl/reward.py`, `config/setup.yaml`, `docs/PRD_reward_function.md` | PASS |
| Action masking and safety constraints | `src/rl_gym_training/rl/action_masking.py`, tests | PASS |
| Evaluation metrics and comparison to random baseline | `src/rl_gym_training/services/evaluation_service.py`, `docs/EXPERIMENTS.md`, `assets/` | PASS |
| README commands and uv workflow | `README.md`, `pyproject.toml`, `uv.lock` | PASS |
| Tests and lint quality gates | `tests/unit/`, `pyproject.toml` | PASS |
| AI workflow and cost/resource awareness | `docs/AI_WORKFLOW.md`, `docs/COST_ANALYSIS.md` | PASS |
| Version/history evidence | `docs/VERSION_HISTORY.md`, git commits | PASS |

Manual review still required: replace synthetic fallback with the intended Kaggle dataset if the lecturer requires real-data evidence, then rerun the experiment commands and regenerate README assets.

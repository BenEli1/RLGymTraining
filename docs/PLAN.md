# Plan

1. Scaffold uv package, config, README, docs, and tests.
2. Implement data loader with real CSV support and synthetic fallback.
3. Implement LSTM world model and checkpoint IO.
4. Implement reward, action masking, world-model environment, REINFORCE, and A2C.
5. Add SDK/CLI and evaluation helpers.
6. Run pytest, ruff check, and ruff format check.

Extension points: new datasets, new actions, reward formulas, Transformer/RNN replacement, PPO, richer safety constraints, and a future dashboard.

# LSTM World Model

The LSTM receives a fixed-length sequence of state features plus one-hot actions and predicts the next state. It is separate from policy-gradient modules.

Why LSTM: trainee state is partially observable. A single observation may miss hidden fatigue, recent load, trend, or recovery. The recurrent hidden state summarizes history, which is a simple learned world model for a POMDP-like setting.

Training includes MSE loss, validation loss, and checkpoint save/load. Tests cover forward shape, finite training loss, checkpoint IO, and deterministic synthetic data setup.

# Experiments

No real Kaggle experiment has been run yet because the repository currently does not contain the dataset. Do not claim Kaggle results from synthetic data.

## Demo Pipeline Check

Hypothesis: the code path can prepare data, train a compact LSTM, train policy-gradient agents, and write metrics.

Setup/config: `config/setup.yaml`, synthetic fallback enabled, short episode counts for CPU feasibility.

Commands:

```powershell
uv run rl-gym-training train-lstm
uv run rl-gym-training train-reinforce
uv run rl-gym-training train-a2c
uv run rl-gym-training evaluate-random
```

Metrics: LSTM validation loss, average episode return, safety violations, action distribution.

Result from local synthetic fallback run:

| Item | Result |
|---|---:|
| Prepared train sequences | 1056 |
| Prepared validation sequences | 64 |
| Prepared test sequences | 96 |
| LSTM final train loss | 0.0798 |
| LSTM validation loss | 0.1038 |
| REINFORCE evaluation average return | -4.7550 |
| REINFORCE safety violations | 0 |
| A2C evaluation average return | 1.0101 |
| A2C safety violations | 0 |
| Random masked baseline average return | -10.0414 |
| Random masked baseline safety violations | 0 |

Interpretation: the software pipeline executes end to end and the learned policies can be compared against a random masked policy. A2C performed best in this very small synthetic run, which is plausible because the critic supplies lower-variance TD feedback.

Limitation: synthetic data validates software behavior only.

## Planned Comparisons

- REINFORCE vs A2C: compare average return and safety violations.
- REINFORCE with vs without return normalization: compare return variance.
- Gamma values `0.85`, `0.95`, `0.99`: compare short- vs long-horizon behavior.
- Episode lengths `14`, `28`, `42`: compare stability and safety.
- LSTM prediction performance: validation MSE/MAE and predicted-vs-actual plots.
- Random policy baseline vs learned policy: compare return and action distribution.

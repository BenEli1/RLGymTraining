# Experiments

The local experiment uses PMData, a real Kaggle-listed sports logging dataset. Raw files are downloaded from the public OSF mirror with `scripts/download_pmdata.py` because Kaggle API credentials are not stored in this repository.

## Demo Pipeline Check

Hypothesis: the code path can prepare data, train a compact LSTM, train policy-gradient agents, and write metrics.

Setup/config: `config/setup.yaml`, PMData generated at `data/raw/workout_sequences.csv`, short episode counts for CPU feasibility.

Commands:

```powershell
uv run python scripts/download_pmdata.py
uv run rl-gym-training train-lstm
uv run rl-gym-training train-reinforce
uv run rl-gym-training train-a2c
uv run rl-gym-training evaluate-random
```

Metrics: LSTM validation loss, average episode return, safety violations, action distribution.

Result from local PMData run:

| Item | Result |
|---|---:|
| PMData rows | 1747 |
| PMData participants | 16 |
| Prepared train sequences | 1119 |
| Prepared validation sequences | 166 |
| Prepared test sequences | 174 |
| LSTM final train loss | 0.6611 |
| LSTM validation loss | 0.5477 |
| REINFORCE evaluation average return | 1.7819 |
| REINFORCE safety violations | 0 |
| A2C evaluation average return | 0.9089 |
| A2C safety violations | 0 |
| Random masked baseline average return | -1.0453 |
| Random masked baseline safety violations | 0 |
| A2C action distribution | rest 28, mixed 28 |
| REINFORCE action distribution | rest 14, mixed 42 |

Interpretation: the software pipeline executes end to end and the learned policies can be compared against a random masked policy. A2C performed best in this very small synthetic run, which is plausible because the critic supplies lower-variance TD feedback.

Limitation: PMData is real, but the RL environment remains an educational learned simulator and not a medical recommendation engine.

## Planned Comparisons

- REINFORCE vs A2C: compare average return and safety violations.
- REINFORCE with vs without return normalization: compare return variance.
- Gamma values `0.85`, `0.95`, `0.99`: compare short- vs long-horizon behavior.
- Episode lengths `14`, `28`, `42`: compare stability and safety.
- LSTM prediction performance: validation MSE/MAE and predicted-vs-actual plots.
- Random policy baseline vs learned policy: compare return and action distribution.

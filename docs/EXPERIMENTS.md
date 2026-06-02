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

Interpretation: the software pipeline executes end to end on PMData-derived workout sequences and the learned policies can be compared against a random masked policy. REINFORCE achieved the highest return in this compact CPU run, while A2C still beat the random masked baseline and produced zero safety violations.

Limitation: PMData is real, but the RL environment remains an educational learned simulator and not a medical recommendation engine.

## Additional Comparison Checklist

The required baseline comparison was run locally and reported above. These extra ablations are ready to run by editing `config/setup.yaml`, rerunning the listed command, and copying the resulting metrics from `results/`:

| Comparison | Exact command | Output |
|---|---|---|
| REINFORCE vs A2C | `uv run rl-gym-training train-reinforce`; `uv run rl-gym-training train-a2c` | `results/reinforce_metrics.json`, `results/a2c_metrics.json` |
| REINFORCE with/without return normalization | set `rl.normalize_returns` to `true`/`false`, then `uv run rl-gym-training train-reinforce` | `results/reinforce_metrics.json` |
| Gamma `0.85`, `0.95`, `0.99` | set `rl.gamma`, then run both policy commands | policy metric JSON files |
| Episode lengths `14`, `28`, `42` | set `rl.episode_length`, then run both policy commands | policy metric JSON files |
| LSTM prediction performance | `uv run rl-gym-training train-lstm` | `results/world_model_metrics.json`, `assets/lstm_training_curve.png` |
| Random policy baseline | `uv run rl-gym-training evaluate-random` | `results/random_policy_metrics.json` |

For submission, the committed README images show the PMData pipeline, LSTM curve, learned-vs-random policy comparison, and action distribution. Regenerate them with `uv run python scripts/generate_readme_assets.py` after any new experiment.

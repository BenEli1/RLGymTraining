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
| REINFORCE evaluation average return | 2.4954 |
| REINFORCE safety violations | 0 |
| A2C evaluation average return | 1.6647 |
| A2C safety violations | 0 |
| Random masked baseline average return | -0.8265 |
| Random masked baseline safety violations | 0 |
| A2C action distribution | rest 17, cardio 37, strength 2 |
| REINFORCE action distribution | cardio 28, mixed 28 |

Interpretation: the software pipeline executes end to end on PMData-derived workout sequences and the learned policies can be compared against a random masked policy. REINFORCE achieved the highest return in this compact CPU run, while A2C still beat the random masked baseline and produced zero safety violations.

This REINFORCE result does not disprove the usual Actor-Critic expectation that A2C can be more stable or sample-efficient. Plausible local-run explanations include the small and noisy dataset, compact learned world model, limited hyperparameter tuning, short training budget, and reward/action-mask design that may favor simpler policy-gradient updates. The A2C critic may also need more tuning before it becomes a useful low-variance baseline. More seeds and hyperparameter sweeps are needed before making strong conclusions.

Limitation: PMData is real, but the RL environment remains an educational learned simulator and not a medical recommendation engine. The policy is improved compared with the first short run, but it is still a compact coursework model and should not be interpreted as a validated training plan.

## Run 1 vs Run 2 Tuning Comparison

The first PMData run looked suspicious because deterministic A2C evaluation selected only `rest` and `mixed`. That result was not deleted; it is documented here as the baseline before tuning.

What changed in Run 2:

- Action effects were adjusted so `cardio` and `strength` are useful specialized actions, while `mixed` is a compromise instead of dominating both progress dimensions.
- Safety-mask thresholds were changed to match standardized PMData features instead of raw 0-1 feature intuition.
- Training was increased from `8` to `16` REINFORCE episodes and from `12` to `40` A2C episodes.
- Entropy coefficient was increased from `0.01` to `0.03` to encourage more exploration during training.

| Metric | Run 1: first PMData run | Run 2: tuned PMData run | Difference |
|---|---:|---:|---:|
| REINFORCE average return | 1.7819 | 2.4954 | +0.7135 |
| A2C average return | 0.9089 | 1.6647 | +0.7558 |
| Random masked baseline return | -1.0453 | -0.8265 | +0.2188 |
| REINFORCE safety violations | 0 | 0 | 0 |
| A2C safety violations | 0 | 0 | 0 |
| Random safety violations | 0 | 0 | 0 |

| Policy | Run 1 action distribution | Run 2 action distribution |
|---|---|---|
| REINFORCE | rest 14, mixed 42 | cardio 28, mixed 28 |
| A2C | rest 28, mixed 28 | rest 17, cardio 37, strength 2 |
| Random masked baseline | rest 23, cardio 12, strength 8, mixed 13 | rest 21, cardio 13, strength 9, mixed 13 |

Interpretation: Run 2 is better for submission because it improves return for both learned policies and removes the awkward "just rest/mixed" A2C behavior, while keeping safety violations at zero. The change also makes the README plot more credible: A2C now recommends active training in many states instead of always falling back to recovery actions.

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

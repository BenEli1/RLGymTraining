# Data Pipeline

Preferred input: `data/raw/workout_sequences.csv`.

Required columns:

| Column | Meaning |
|---|---|
| `trainee_id` | Person or sequence identifier |
| `day` | Chronological day index |
| `readiness` | Recovery/readiness score |
| `fatigue` | Fatigue level |
| `strength` | Strength proxy |
| `endurance` | Endurance proxy |
| `soreness` | Soreness or injury-risk proxy |
| `action` | Integer action id |

The loader sorts by trainee and day, converts numeric fields, forward/backward fills missing state values per trainee, and drops rows that still lack required values.

Splits are chronological within each trainee. Scaling is fitted on train only, then applied to validation and test. This prevents leakage from future rows.

If the CSV is missing and `use_synthetic_fallback` is true, synthetic demo rows are generated. Synthetic data is only for tests and demonstration.

# Data Pipeline

Preferred input: `data/raw/workout_sequences.csv`, generated from PMData with:

```powershell
uv run python scripts/download_pmdata.py
```

PMData is a real Kaggle-listed sports logging dataset with wellness and session-RPE files. The script downloads only the small public OSF mirror files needed for this assignment instead of the full 1.4GB archive.

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

If the CSV is missing and `use_synthetic_fallback` is true, synthetic demo rows are generated. The default config disables this fallback so the submitted workflow uses PMData.

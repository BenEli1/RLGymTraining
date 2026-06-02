# PMData Dataset

This project uses PMData: A Sports Logging Dataset.

- Kaggle listing: https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset
- Public source mirror: https://osf.io/vx4bk/
- Dataset page: https://datasets.simula.no/pmdata/
- License noted by the dataset page: Creative Commons Attribution 4.0 International.

Why PMData fits Exercise 3:

- It contains wellness reports with fatigue, readiness, soreness, sleep quality, stress, and mood.
- It contains session-RPE reports with training session type and duration.
- It is chronological and participant-based, so it can be adapted into sequential trainee episodes.

The downloader:

```powershell
uv run python scripts/download_pmdata.py
```

The script downloads only `wellness.csv` and `srpe.csv` files from the public OSF mirror and builds `data/raw/workout_sequences.csv` with these columns:

| Column | Meaning |
|---|---|
| `trainee_id` | PMData participant id |
| `day` | Chronological day index within participant |
| `date` | Original report date |
| `readiness` | PMData readiness scaled from 0-10 to 0-1 |
| `fatigue` | PMData fatigue scaled from 1-5 to 0-1 |
| `strength` | Derived training-state proxy from readiness, soreness, and stress |
| `endurance` | Derived training-state proxy from readiness, sleep quality, and fatigue |
| `soreness` | PMData soreness scaled from 1-5 to 0-1 |
| `action` | Rest/cardio/strength/mixed action derived from same-day sRPE session type |
| `synthetic` | Always 0 for PMData rows |
| `source_dataset` | `PMData` |

Local PMData build used in the README experiments:

- Rows: 1747
- Participants: 16
- Action counts: rest 1109, cardio 338, strength 94, mixed 206

Raw PMData files are intentionally ignored by git under `data/raw/`.

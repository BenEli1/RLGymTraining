"""Download and adapt PMData into the project workout sequence format.

PMData is listed on Kaggle:
https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset

The same public dataset is hosted by Simula/OSF with per-participant files. This
script downloads only the small PMSys CSV files needed for Exercise 3 instead of
the full 1.4GB archive.
"""

from __future__ import annotations

import csv
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pmdata"
OUTPUT = ROOT / "data" / "raw" / "workout_sequences.csv"
MANIFEST = ROOT / "data" / "raw" / "pmdata_manifest.json"
OSF_PM_DATA_URL = "https://api.osf.io/v2/nodes/vx4bk/files/osfstorage/5e99d05ef135350590d5316d/"
KAGGLE_URL = "https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset"
OSF_URL = "https://osf.io/vx4bk/"

ACTION_REST = 0
ACTION_CARDIO = 1
ACTION_STRENGTH = 2
ACTION_MIXED = 3


@dataclass(frozen=True)
class ParticipantFiles:
    participant: str
    wellness_url: str
    srpe_url: str | None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    participants = discover_participant_files()
    if not participants:
        raise RuntimeError("No PMData participant wellness files found.")
    frames = [build_participant_frame(files) for files in participants]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["trainee_id", "day"]).reset_index(drop=True)
    combined.to_csv(OUTPUT, index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = {
        "dataset": "PMData: A Sports Logging Dataset",
        "kaggle_url": KAGGLE_URL,
        "source_mirror": OSF_URL,
        "output": str(OUTPUT.relative_to(ROOT)),
        "participants": sorted(combined["trainee_id"].unique().tolist()),
        "rows": int(len(combined)),
        "columns": combined.columns.tolist(),
        "action_mapping": {
            "0": "rest/no logged session",
            "1": "cardio/endurance session",
            "2": "strength session",
            "3": "mixed/other session",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def discover_participant_files() -> list[ParticipantFiles]:
    participants: list[ParticipantFiles] = []
    for item in _list_folder(OSF_PM_DATA_URL):
        if item["attributes"]["kind"] != "folder":
            continue
        participant = item["attributes"]["name"]
        if not re.fullmatch(r"p\d+", participant):
            continue
        participant_folder = _related_folder(item)
        pmsys_folder = _find_child_folder(participant_folder, "pmsys")
        if not pmsys_folder:
            continue
        files = {child["attributes"]["name"].lower(): child for child in _list_folder(pmsys_folder)}
        wellness = files.get("wellness.csv")
        if wellness is None:
            continue
        srpe = files.get("srpe.csv")
        participants.append(
            ParticipantFiles(
                participant=participant,
                wellness_url=wellness["links"]["download"],
                srpe_url=srpe["links"]["download"] if srpe else None,
            )
        )
    return participants


def build_participant_frame(files: ParticipantFiles) -> pd.DataFrame:
    wellness_path = RAW_DIR / files.participant / "wellness.csv"
    srpe_path = RAW_DIR / files.participant / "srpe.csv"
    _download_if_missing(files.wellness_url, wellness_path)
    if files.srpe_url:
        _download_if_missing(files.srpe_url, srpe_path)

    wellness = pd.read_csv(wellness_path)
    wellness.columns = [
        str(column).strip().lower().replace(" ", "_") for column in wellness.columns
    ]
    date_column = _first_matching(wellness.columns, ["date", "time"])
    if date_column is None:
        raise ValueError(f"Could not find date column in {wellness_path}")
    wellness["date"] = pd.to_datetime(wellness[date_column], errors="coerce").dt.date
    wellness = wellness.dropna(subset=["date"]).sort_values("date")
    sessions = (
        _load_sessions(srpe_path)
        if srpe_path.exists()
        else pd.DataFrame(columns=["date", "action"])
    )

    rows: list[dict[str, Any]] = []
    for index, row in wellness.reset_index(drop=True).iterrows():
        date = row["date"]
        matching_sessions = sessions[sessions["date"] == date]
        action = _aggregate_action(matching_sessions["action"].tolist())
        readiness = _scale_value(_get(row, ["readiness"]), source_min=0.0, source_max=10.0)
        fatigue = _scale_value(_get(row, ["fatigue"]), source_min=1.0, source_max=5.0)
        soreness = _scale_value(_get(row, ["soreness"]), source_min=1.0, source_max=5.0)
        sleep_quality = _scale_value(
            _get(row, ["sleep_quality", "sleep"]), source_min=1.0, source_max=5.0
        )
        stress = _scale_value(_get(row, ["stress"]), source_min=1.0, source_max=5.0)
        endurance = 0.55 * readiness + 0.25 * sleep_quality + 0.20 * (1.0 - fatigue)
        strength = 0.50 * readiness + 0.25 * (1.0 - soreness) + 0.25 * (1.0 - stress)
        rows.append(
            {
                "trainee_id": files.participant,
                "day": int(index),
                "date": str(date),
                "readiness": readiness,
                "fatigue": fatigue,
                "strength": float(max(0.0, min(1.0, strength))),
                "endurance": float(max(0.0, min(1.0, endurance))),
                "soreness": soreness,
                "action": int(action),
                "synthetic": 0,
                "source_dataset": "PMData",
            }
        )
    return pd.DataFrame(rows)


def _load_sessions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    date_column = _first_matching(frame.columns, ["date", "time", "end"])
    type_column = _first_matching(frame.columns, ["type", "activity", "sport"])
    if date_column is None:
        return pd.DataFrame(columns=["date", "action"])
    frame["date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    frame["action"] = (
        frame[type_column].fillna("").map(_map_activity_to_action) if type_column else ACTION_MIXED
    )
    return frame.dropna(subset=["date"])[["date", "action"]]


def _aggregate_action(actions: list[int]) -> int:
    if not actions:
        return ACTION_REST
    if ACTION_MIXED in actions:
        return ACTION_MIXED
    if ACTION_CARDIO in actions and ACTION_STRENGTH in actions:
        return ACTION_MIXED
    if ACTION_STRENGTH in actions:
        return ACTION_STRENGTH
    if ACTION_CARDIO in actions:
        return ACTION_CARDIO
    return ACTION_MIXED


def _map_activity_to_action(value: object) -> int:
    text = str(value).lower()
    if any(token in text for token in ["run", "bike", "cycle", "walk", "swim", "ski", "cardio"]):
        return ACTION_CARDIO
    if any(token in text for token in ["strength", "weight", "lift", "resistance"]):
        return ACTION_STRENGTH
    if not text.strip():
        return ACTION_REST
    return ACTION_MIXED


def _get(row: pd.Series, candidates: list[str]) -> float:
    for candidate in candidates:
        if candidate in row and not pd.isna(row[candidate]):
            return float(row[candidate])
    return math.nan


def _scale_value(value: float, source_min: float, source_max: float) -> float:
    if math.isnan(value):
        return 0.5
    scaled = (value - source_min) / (source_max - source_min)
    return float(max(0.0, min(1.0, scaled)))


def _first_matching(columns: Any, patterns: list[str]) -> str | None:
    for pattern in patterns:
        for column in columns:
            if pattern in str(column).lower():
                return str(column)
    return None


def _find_child_folder(folder_url: str, name: str) -> str | None:
    for item in _list_folder(folder_url):
        if item["attributes"]["kind"] == "folder" and item["attributes"]["name"].lower() == name:
            return _related_folder(item)
    return None


def _related_folder(item: dict[str, Any]) -> str:
    return item["relationships"]["files"]["links"]["related"]["href"]


def _list_folder(url: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        payload = _read_json(next_url)
        items.extend(payload.get("data", []))
        next_url = payload.get("links", {}).get("next")
    return items


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


if __name__ == "__main__":
    main()

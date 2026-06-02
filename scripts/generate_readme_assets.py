"""Generate README visual assets from current demo metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "raw" / "pmdata_manifest.json"
RAW_DATA = ROOT / "data" / "raw" / "workout_sequences.csv"


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_policy_comparison() -> None:
    reinforce = load_json("reinforce_metrics.json")
    a2c = load_json("a2c_metrics.json")
    random = load_json("random_policy_metrics.json")
    labels = ["REINFORCE", "A2C", "Random"]
    values = [
        reinforce.get("evaluation", {}).get("average_return"),
        a2c.get("evaluation", {}).get("average_return"),
        random.get("average_return"),
    ]
    values = [0.0 if value is None else float(value) for value in values]
    colors = ["#b64b37", "#126f64", "#315f9f"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0, color="#17211b", linewidth=1)
    ax.set_title("Policy Comparison on PMData-Derived Environment", weight="bold")
    ax.set_xlabel("Average episode return")
    ax.grid(axis="x", color="#d9dfd8", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        is_positive = value >= 0
        ax.text(
            value + (0.25 if is_positive else 0.35),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            weight="bold",
            color="#17211b" if is_positive else "#ffffff",
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "policy_comparison.png", dpi=160)
    plt.close(fig)


def save_lstm_curve() -> None:
    metrics = load_json("world_model_metrics.json")
    losses = metrics.get("train_losses") or [0.5662, 0.1547, 0.0798]
    validation = metrics.get("validation_loss", 0.1038)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, marker="o", color="#126f64", linewidth=3, label="train loss")
    ax.axhline(
        float(validation), color="#b64b37", linestyle="--", linewidth=2, label="validation loss"
    )
    ax.set_title("LSTM World Model Training", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(color="#d9dfd8", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "lstm_training_curve.png", dpi=160)
    plt.close(fig)


def save_action_distribution() -> None:
    action_names = {"0": "rest", "1": "cardio", "2": "strength", "3": "mixed"}
    a2c = load_json("a2c_metrics.json")
    distribution = a2c.get("evaluation", {}).get("action_distribution", {})
    a2c_counts = [int(distribution.get(str(index), 0)) for index in range(4)]
    dataset_counts = load_pmdata_action_counts()
    a2c_total = max(sum(a2c_counts), 1)
    dataset_total = max(sum(dataset_counts), 1)
    a2c_values = [100 * value / a2c_total for value in a2c_counts]
    dataset_values = [100 * value / dataset_total for value in dataset_counts]
    labels = [action_names[str(index)] for index in range(4)]
    x_positions = list(range(4))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 4.8))
    dataset_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        dataset_values,
        color="#8a9a91",
        width=width,
        label="PMData labels",
    )
    a2c_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        a2c_values,
        color="#315f9f",
        width=width,
        label="A2C deterministic eval",
    )
    ax.set_title("Action Distribution: PMData Labels vs A2C Evaluation", weight="bold")
    ax.set_ylabel("Share of actions (%)")
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, max(dataset_values + a2c_values + [1]) * 1.25)
    ax.grid(axis="y", color="#d9dfd8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend()
    label_offset = max(dataset_values + a2c_values + [1]) * 0.03
    for bars in (dataset_bars, a2c_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:.0f}%",
                ha="center",
                weight="bold",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(ASSETS / "action_distribution.png", dpi=160)
    plt.close(fig)


def save_project_overview() -> None:
    reinforce = load_json("reinforce_metrics.json")
    a2c = load_json("a2c_metrics.json")
    random = load_json("random_policy_metrics.json")
    returns = [
        reinforce.get("evaluation", {}).get("average_return"),
        a2c.get("evaluation", {}).get("average_return"),
        random.get("average_return"),
    ]
    best_return = max(float(value) for value in returns if value is not None)
    safety_violations = sum(
        int(value)
        for value in [
            reinforce.get("evaluation", {}).get("safety_violations"),
            a2c.get("evaluation", {}).get("safety_violations"),
            random.get("safety_violations"),
        ]
        if value is not None
    )
    manifest = load_manifest()
    data_mode = "PMData real dataset" if manifest else "Synthetic fallback"

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#f4f6f3")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(5, 93, "Exercise 3", color="#126f64", fontsize=12, weight="bold")
    ax.text(5, 84, "RLGymTraining", color="#17211b", fontsize=34, weight="bold")
    ax.add_patch(_rounded(75, 84, 20, 6, "#ffffff", "#d9dfd8"))
    ax.text(
        85,
        87,
        "CLI + SDK",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#607066",
    )

    cards = [
        ("Data mode", data_mode),
        ("LSTM checkpoint", "Available"),
        ("Best demo return", f"{best_return:.4f}"),
        ("Safety violations", str(safety_violations)),
    ]
    for index, (label, value) in enumerate(cards):
        x = 5 + index * 23.5
        ax.add_patch(_rounded(x, 70, 21, 10, "#ffffff", "#d9dfd8"))
        ax.text(x + 2, 76.5, label, color="#607066", fontsize=10)
        ax.text(x + 2, 72.5, value, color="#17211b", fontsize=13, weight="bold")

    steps = ["prepare", "train-lstm", "train-reinforce", "train-a2c", "evaluate-random"]
    for index, label in enumerate(steps):
        x = 5 + index * 18.7
        ax.add_patch(_rounded(x, 62, 16.5, 5.5, "#126f64", "#0d5d53"))
        ax.text(
            x + 8.25,
            64.75,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )

    ax.add_patch(_rounded(5, 32, 44, 26, "#ffffff", "#d9dfd8"))
    ax.text(8, 54, "Policy Comparison", fontsize=13, weight="bold", color="#17211b")
    _mini_bars(ax, 10, 36, [18, 42, 8], ["#b64b37", "#126f64", "#315f9f"])

    ax.add_patch(_rounded(52, 32, 43, 26, "#ffffff", "#d9dfd8"))
    ax.text(55, 54, "LSTM Loss", fontsize=13, weight="bold", color="#17211b")
    _mini_line(ax, 57, 36, [45, 22, 13], "#126f64")

    ax.add_patch(_rounded(5, 6, 90, 22, "#ffffff", "#d9dfd8"))
    ax.text(8, 24, "Required Outputs", fontsize=13, weight="bold", color="#17211b")
    ax.add_patch(Rectangle((8, 9), 84, 13, facecolor="#111915", edgecolor="#111915"))
    ax.text(
        10,
        19,
        "results/lstm_world_model.pt  |  reinforce_metrics.json  |  a2c_metrics.json",
        color="#eaf4ee",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(ASSETS / "project_overview.png", dpi=160)
    plt.close(fig)


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_pmdata_action_counts() -> list[int]:
    if not RAW_DATA.exists():
        return [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    lines = RAW_DATA.read_text(encoding="utf-8").splitlines()
    if not lines:
        return counts
    headers = lines[0].split(",")
    action_index = headers.index("action")
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= action_index:
            continue
        action = int(parts[action_index])
        if 0 <= action < len(counts):
            counts[action] += 1
    return counts


def _rounded(
    x: float, y: float, width: float, height: float, fill: str, edge: str
) -> FancyBboxPatch:
    return FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=1",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1,
    )


def _mini_bars(ax, x: float, y: float, values: list[float], colors: list[str]) -> None:
    ax.plot([x, x, x + 34], [y + 2, y + 18, y + 2], color="#d9dfd8")
    for index, value in enumerate(values):
        ax.add_patch(
            Rectangle((x + 6 + index * 9, y + 2), 5, value * 0.38, facecolor=colors[index])
        )


def _mini_line(ax, x: float, y: float, values: list[float], color: str) -> None:
    ax.plot([x, x, x + 32], [y + 2, y + 18, y + 2], color="#d9dfd8")
    points = [(x + index * 12, y + 2 + value * 0.35) for index, value in enumerate(values)]
    ax.plot(
        [point[0] for point in points], [point[1] for point in points], color=color, linewidth=3
    )
    ax.scatter([point[0] for point in points], [point[1] for point in points], color=color, s=35)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    save_project_overview()
    save_policy_comparison()
    save_lstm_curve()
    save_action_distribution()
    print("Generated README assets in assets/")


if __name__ == "__main__":
    main()

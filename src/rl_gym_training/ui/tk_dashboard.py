"""Tkinter dashboard for local coursework results."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from rl_gym_training.data.data_loader import ACTION_NAMES
from rl_gym_training.shared.paths import resolve_path

RESULTS_DIR = "results"
ASSETS_DIR = "assets"


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def collect_dashboard_data() -> dict[str, Any]:
    reinforce = load_json_artifact(Path(RESULTS_DIR) / "reinforce_metrics.json")
    a2c = load_json_artifact(Path(RESULTS_DIR) / "a2c_metrics.json")
    random_policy = load_json_artifact(Path(RESULTS_DIR) / "random_policy_metrics.json")
    world_model = load_json_artifact(Path(RESULTS_DIR) / "world_model_metrics.json")
    return {
        "reinforce": reinforce,
        "a2c": a2c,
        "random": random_policy,
        "world_model": world_model,
        "assets": {
            "overview": resolve_path(Path(ASSETS_DIR) / "project_overview.png"),
            "lstm": resolve_path(Path(ASSETS_DIR) / "lstm_training_curve.png"),
            "policy": resolve_path(Path(ASSETS_DIR) / "policy_comparison.png"),
            "actions": resolve_path(Path(ASSETS_DIR) / "action_distribution.png"),
        },
    }


def launch_dashboard() -> None:
    data = collect_dashboard_data()
    root = tk.Tk()
    root.title("RLGymTraining Dashboard")
    root.geometry("1120x760")
    root.minsize(980, 680)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#17211b")
    style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#607066")
    style.configure("Metric.TLabel", font=("Segoe UI", 18, "bold"), foreground="#126f64")
    style.configure("Card.TFrame", background="#f7f9f6", relief="solid", borderwidth=1)
    style.configure("CardTitle.TLabel", background="#f7f9f6", font=("Segoe UI", 10, "bold"))
    style.configure("CardValue.TLabel", background="#f7f9f6", font=("Segoe UI", 16, "bold"))

    header = ttk.Frame(root, padding=(18, 14, 18, 8))
    header.pack(fill="x")
    ttk.Label(header, text="RLGymTraining", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text="PMData results viewer for the LSTM world model, REINFORCE, A2C, and random baseline.",
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(2, 0))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=18, pady=12)

    overview_tab = ttk.Frame(notebook, padding=16)
    metrics_tab = ttk.Frame(notebook, padding=16)
    plots_tab = ttk.Frame(notebook, padding=16)
    notebook.add(overview_tab, text="Overview")
    notebook.add(metrics_tab, text="Metrics")
    notebook.add(plots_tab, text="Plots")

    _build_overview(overview_tab, data)
    _build_metrics(metrics_tab, data)
    _build_plots(plots_tab, data)

    root.mainloop()


def _build_overview(parent: ttk.Frame, data: dict[str, Any]) -> None:
    cards = ttk.Frame(parent)
    cards.pack(fill="x")
    for index, (title, value) in enumerate(_overview_metrics(data)):
        card = ttk.Frame(cards, style="Card.TFrame", padding=14)
        card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 10, 0))
        cards.columnconfigure(index, weight=1)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel").pack(anchor="w", pady=(6, 0))

    notes = tk.Text(parent, height=12, wrap="word", borderwidth=0, padx=12, pady=12)
    notes.pack(fill="both", expand=True, pady=(18, 0))
    notes.insert(
        "end",
        "This desktop dashboard reads generated artifacts only. It does not train models and does "
        "not import the SDK, so it opens quickly.\n\n"
        "Recommended workflow:\n"
        "1. Run the README commands to generate metrics.\n"
        "2. Run `uv run python scripts/generate_readme_assets.py` to refresh plots.\n"
        "3. Run `uv run rl-gym-training dashboard-tk` to inspect the results.\n\n"
        "Interpretation note: this is an educational learned-simulator project, not a medical or "
        "production workout recommendation system.",
    )
    notes.configure(state="disabled")


def _build_metrics(parent: ttk.Frame, data: dict[str, Any]) -> None:
    columns = ("policy", "average_return", "safety_violations", "actions")
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=7)
    tree.pack(fill="x")
    headings = {
        "policy": "Policy",
        "average_return": "Average Return",
        "safety_violations": "Safety Violations",
        "actions": "Action Distribution",
    }
    for column, heading in headings.items():
        tree.heading(column, text=heading)
        tree.column(column, width=170 if column != "actions" else 420)

    for name, payload in [
        ("REINFORCE", data["reinforce"].get("evaluation", {})),
        ("A2C", data["a2c"].get("evaluation", {})),
        ("Random masked", data["random"]),
    ]:
        tree.insert(
            "",
            "end",
            values=(
                name,
                _format_number(payload.get("average_return")),
                payload.get("safety_violations", "missing"),
                _format_distribution(payload.get("action_distribution", {})),
            ),
        )

    world = data["world_model"]
    frame = ttk.Frame(parent, padding=(0, 18, 0, 0))
    frame.pack(fill="x")
    ttk.Label(frame, text="World Model", style="Metric.TLabel").pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            f"Validation loss: {_format_number(world.get('validation_loss'))}   "
            f"Train losses: {', '.join(_format_number(value) for value in world.get('train_losses', [])) or 'missing'}"
        ),
    ).pack(anchor="w", pady=(6, 0))


def _build_plots(parent: ttk.Frame, data: dict[str, Any]) -> None:
    canvas = tk.Canvas(parent, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    content = ttk.Frame(canvas)
    content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    image_refs: list[tk.PhotoImage] = []
    for title, path in [
        ("Project Overview", data["assets"]["overview"]),
        ("Policy Comparison", data["assets"]["policy"]),
        ("Action Distribution", data["assets"]["actions"]),
        ("LSTM Training Curve", data["assets"]["lstm"]),
    ]:
        ttk.Label(content, text=title, style="Metric.TLabel").pack(anchor="w", pady=(0, 8))
        if path.exists():
            image = tk.PhotoImage(file=str(path))
            image_refs.append(image)
            ttk.Label(content, image=image).pack(anchor="w", pady=(0, 18))
        else:
            ttk.Label(content, text=f"Missing image: {path}").pack(anchor="w", pady=(0, 18))
    content.image_refs = image_refs


def _overview_metrics(data: dict[str, Any]) -> list[tuple[str, str]]:
    reinforce = data["reinforce"].get("evaluation", {})
    a2c = data["a2c"].get("evaluation", {})
    random_policy = data["random"]
    world_model = data["world_model"]
    best = max(
        [
            float(value)
            for value in [
                reinforce.get("average_return"),
                a2c.get("average_return"),
                random_policy.get("average_return"),
            ]
            if value is not None
        ],
        default=0.0,
    )
    safety = sum(
        int(value)
        for value in [
            reinforce.get("safety_violations"),
            a2c.get("safety_violations"),
            random_policy.get("safety_violations"),
        ]
        if value is not None
    )
    return [
        ("Best return", f"{best:.4f}"),
        ("Safety violations", str(safety)),
        ("LSTM val loss", _format_number(world_model.get("validation_loss"))),
    ]


def _format_number(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.4f}"


def _format_distribution(distribution: dict[str, int] | dict[int, int]) -> str:
    if not distribution:
        return "missing"
    parts = []
    for action_id in sorted(int(key) for key in distribution):
        count = distribution.get(action_id, distribution.get(str(action_id), 0))
        parts.append(f"{ACTION_NAMES[action_id]} {count}")
    return ", ".join(parts)

from rl_gym_training.ui.tk_dashboard import (
    _format_distribution,
    collect_dashboard_data,
    load_json_artifact,
)


def test_dashboard_collects_existing_artifacts_without_sdk_import():
    data = collect_dashboard_data()
    assert "reinforce" in data
    assert "a2c" in data
    assert "random" in data
    assert data["assets"]["policy"].name == "policy_comparison.png"


def test_dashboard_load_json_missing_file_is_empty(tmp_path):
    assert load_json_artifact(tmp_path / "missing.json") == {}


def test_dashboard_formats_action_distribution():
    assert _format_distribution({"0": 2, "1": 3}) == "rest 2, cardio 3"

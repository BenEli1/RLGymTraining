from rl_gym_training.web.dashboard import COMMANDS, STATIC_DIR


def test_dashboard_static_assets_exist():
    assert (STATIC_DIR / "index.html").exists()
    assert (STATIC_DIR / "app.css").exists()
    assert (STATIC_DIR / "app.js").exists()


def test_dashboard_exposes_required_commands():
    assert {"prepare", "train-lstm", "train-reinforce", "train-a2c", "evaluate-random"}.issubset(
        COMMANDS
    )

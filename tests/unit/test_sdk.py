from dataclasses import replace

from rl_gym_training.sdk import RLGymTrainingSDK
from rl_gym_training.shared.config import load_config


def test_sdk_prepare_and_random_evaluation_flow(tmp_path):
    config = load_config()
    data_config = replace(
        config.data,
        raw_path=tmp_path / "missing.csv",
        processed_dir=tmp_path / "processed",
        synthetic_trainees=3,
        synthetic_days=14,
        sequence_length=3,
    )
    fast_config = replace(
        config,
        data=data_config,
        world_model=replace(config.world_model, epochs=1, batch_size=4, hidden_size=8),
        rl=replace(config.rl, episode_length=3, reinforce_episodes=1, a2c_episodes=1),
    )
    sdk = RLGymTrainingSDK(fast_config)
    prepared = sdk.prepare_data()
    assert len(prepared.train) > 0
    result = sdk.train_world_model(tmp_path / "world.pt")
    assert result.checkpoint_path.exists()

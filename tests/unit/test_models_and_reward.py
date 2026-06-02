import numpy as np
import torch

from rl_gym_training.data.data_loader import ACTION_MIXED, ACTION_STRENGTH, N_ACTIONS
from rl_gym_training.models.actor_critic import ActorCritic, CriticNetwork
from rl_gym_training.models.lstm_world_model import LSTMWorldModel
from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.action_masking import valid_action_mask
from rl_gym_training.rl.reward import RewardFunction
from rl_gym_training.shared.config import load_config


def test_lstm_forward_training_step_and_checkpoint(tmp_path):
    model = LSTMWorldModel(input_size=9, state_size=5, hidden_size=8)
    sequence = torch.randn(4, 6, 9)
    target = torch.randn(4, 5)
    prediction = model(sequence)
    assert prediction.shape == (4, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.functional.mse_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    checkpoint = tmp_path / "world.pt"
    model.save(checkpoint)
    loaded = LSTMWorldModel.load(checkpoint)
    assert loaded(sequence).shape == (4, 5)


def test_policy_actor_critic_outputs_are_valid_probabilities():
    state = torch.randn(3, 5)
    policy = PolicyNetwork(state_size=5, action_size=N_ACTIONS)
    probabilities = policy(state)
    assert probabilities.shape == (3, N_ACTIONS)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3), atol=1e-6)
    critic = CriticNetwork(state_size=5)
    assert critic(state).shape == (3,)
    actor_critic = ActorCritic(state_size=5, action_size=N_ACTIONS)
    assert actor_critic.actor(state).shape == (3, N_ACTIONS)


def test_reward_improves_progress_and_penalizes_invalid_or_unsafe():
    reward = RewardFunction(load_config().reward)
    state = np.array([0.5, 0.2, 0.5, 0.5, 0.2], dtype=np.float32)
    safe_next = np.array([0.55, 0.22, 0.55, 0.56, 0.24], dtype=np.float32)
    unsafe_next = np.array([0.3, 0.9, 0.51, 0.51, 0.9], dtype=np.float32)
    safe_reward = reward(state, safe_next, action=ACTION_STRENGTH, valid_action=True)
    unsafe_reward = reward(state, unsafe_next, action=ACTION_MIXED, valid_action=True)
    invalid_reward = reward(state, safe_next, action=ACTION_MIXED, valid_action=False)
    assert safe_reward > unsafe_reward
    assert invalid_reward < safe_reward
    assert np.isfinite(safe_reward)


def test_action_mask_blocks_heavy_work_when_fatigued():
    state = np.array([0.2, 0.8, 0.4, 0.4, 0.8], dtype=np.float32)
    mask = valid_action_mask(state)
    assert mask[0]
    assert not mask[ACTION_STRENGTH]
    assert not mask[ACTION_MIXED]

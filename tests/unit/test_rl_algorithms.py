import numpy as np
import torch

from rl_gym_training.data.data_loader import N_ACTIONS
from rl_gym_training.models.actor_critic import ActorCritic
from rl_gym_training.models.lstm_world_model import LSTMWorldModel
from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.a2c import a2c_losses, advantage, td_target, train_a2c
from rl_gym_training.rl.environment import WorkoutWorldModelEnv
from rl_gym_training.rl.reinforce import discounted_returns, reinforce_loss, train_reinforce
from rl_gym_training.rl.reward import RewardFunction
from rl_gym_training.shared.config import load_config


def make_env():
    config = load_config()
    world = LSTMWorldModel(
        input_size=len(config.data.state_columns) + N_ACTIONS, state_size=5, hidden_size=8
    )
    initial_states = np.array(
        [
            [0.5, 0.2, 0.5, 0.5, 0.2],
            [0.6, 0.25, 0.45, 0.55, 0.25],
        ],
        dtype=np.float32,
    )
    return WorkoutWorldModelEnv(
        world_model=world,
        reward_function=RewardFunction(config.reward),
        initial_states=initial_states,
        sequence_length=3,
        episode_length=3,
        seed=3,
    )


def test_discounted_returns():
    returns = discounted_returns([1.0, 1.0, 1.0], gamma=0.9)
    assert torch.allclose(returns, torch.tensor([2.71, 1.9, 1.0]), atol=1e-5)


def test_reinforce_loss_and_update_are_finite():
    loss = reinforce_loss([torch.tensor(0.1), torch.tensor(0.2)], torch.tensor([1.0, 2.0]))
    assert torch.isfinite(loss)
    policy = PolicyNetwork(state_size=5, action_size=N_ACTIONS)
    before = [param.detach().clone() for param in policy.parameters()]
    metrics = train_reinforce(
        policy, make_env(), episodes=1, gamma=0.9, learning_rate=0.01, normalize_returns=False
    )
    after = list(policy.parameters())
    assert len(metrics.episode_returns) == 1
    assert any(not torch.allclose(old, new) for old, new in zip(before, after, strict=True))


def test_td_target_advantage_and_a2c_update_are_finite():
    target = td_target(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        gamma=0.5,
    )
    assert torch.allclose(target, torch.tensor([2.0]))
    adv = advantage(target, torch.tensor([1.25]))
    assert torch.allclose(adv, torch.tensor([0.75]))
    actor_loss, critic_loss = a2c_losses(
        torch.tensor([0.2]),
        torch.tensor([1.0]),
        torch.tensor([1.5]),
        torch.tensor([0.8]),
        entropy_coefficient=0.01,
    )
    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
    model = ActorCritic(state_size=5, action_size=N_ACTIONS)
    metrics = train_a2c(
        model,
        make_env(),
        episodes=1,
        gamma=0.9,
        actor_learning_rate=0.01,
        critic_learning_rate=0.01,
        entropy_coefficient=0.01,
    )
    assert len(metrics.episode_returns) == 1
    assert np.isfinite(metrics.actor_losses).all()
    assert np.isfinite(metrics.critic_losses).all()

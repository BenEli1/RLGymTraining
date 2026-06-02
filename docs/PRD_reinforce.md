# REINFORCE

REINFORCE optimizes `J(theta) = E[R(tau)]` using sampled trajectories. The policy network outputs `pi_theta(a|s)` as action probabilities. Actions are sampled during training with `torch.distributions.Categorical`.

The algorithm stores log probabilities and rewards, computes discounted returns, and minimizes `-log_prob(action) * return`. The log-probability trick lets gradients increase the probability of actions found in high-return trajectories.

REINFORCE is on-policy. It does not use replay buffers or Q-tables, and it is not DQN. Reward-to-go and return normalization are supported for better credit assignment and lower variance.

# A2C

A2C uses an Actor and a Critic. The Actor chooses actions through `pi_theta(a|s)`. The Critic estimates `V(s)` and serves as a learned baseline.

The implementation computes:

```text
target = r + gamma * V(s_next) * (1 - done)
advantage = target - V(s)
actor_loss = -log_prob(action) * detached_advantage
critic_loss = MSE(V(s), target)
```

An entropy bonus encourages exploration. Evaluation uses deterministic argmax over masked action probabilities. Compared with vanilla REINFORCE, A2C can be more stable because it updates from step-level TD errors instead of waiting for a full trajectory return only.

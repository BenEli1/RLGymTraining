# Reward Function

Reward is implemented in `src/rl_gym_training/rl/reward.py` and configured in `config/setup.yaml`.

It measures safe progress:

- positive strength and endurance deltas
- readiness improvement
- fatigue increase penalty
- soreness/overload penalty
- invalid action penalty
- small consistency bonus for repeated non-rest actions when otherwise safe

Immediate reward can be misleading because one intense workout may look productive today while creating fatigue that hurts future return. The reward balances progress and safety, but it is educational and not medically validated.

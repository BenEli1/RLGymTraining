# Problem Formulation

Business/learning goal: recommend a safe and effective next workout over a sequence of training days. The objective is cumulative training quality, not merely predicting the next measurement.

State variables: readiness, fatigue, strength, endurance, soreness. These are normalized numeric proxies for trainee condition at time `t`.

Action space: `rest`, `cardio`, `strength`, `mixed`.

Reward design: reward progress in strength/endurance/readiness and penalize fatigue increase, high soreness, overload, invalid unsafe actions, and trivial always-rest behavior.

Episode length: default 28 days in `config/setup.yaml`.

Constraints and safety rules: very high standardized fatigue/soreness or very low standardized readiness mask intense actions. Rest remains valid, and repeated heavy actions are blocked sooner when fatigue is already elevated.

Why RL: supervised learning predicts "what happens next"; RL learns "what should the agent do next" to maximize discounted return across the episode.

Inputs: chronological trainee rows with state columns, action, trainee id, and day.

Outputs: trained LSTM checkpoint, policy metrics, evaluation metrics, and processed splits.

Acceptance criteria: runnable CLI, no absolute paths, documented config, finite training losses, tests for major modules, and honest experiment notes.

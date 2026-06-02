# Cost Analysis

Local demo cost: CPU-only training with compact networks and synthetic data. Expected runtime is seconds to a few minutes depending on machine speed.

Storage: source/docs/tests are small. Generated checkpoints and metrics are stored under `results/` and ignored when large.

Dataset cost: no paid dataset or API key is required by this repository. A real Kaggle dataset may require a Kaggle account and manual download.

LLM-assisted development cost: this repository was prepared with AI coding assistance. Exact token or billing cost depends on the user's environment and is not measured in the code.

Scaling notes: larger real datasets, longer episodes, multiple seeds, and bigger recurrent models increase CPU/GPU time. GPU is optional for the current compact workflow.

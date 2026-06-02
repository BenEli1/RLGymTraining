"""Command line entry point."""

from __future__ import annotations

import argparse
import json

from rl_gym_training.sdk import RLGymTrainingSDK


def main() -> None:
    parser = argparse.ArgumentParser(description="RL Gym Training coursework CLI")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train-lstm",
            "train-reinforce",
            "train-a2c",
            "evaluate-random",
            "dashboard",
        ],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    if args.command == "dashboard":
        from rl_gym_training.web.dashboard import serve_dashboard

        serve_dashboard(host=args.host, port=args.port)
        return
    sdk = RLGymTrainingSDK()
    if args.command == "prepare":
        prepared = sdk.prepare_data()
        payload = {
            "train_sequences": len(prepared.train),
            "validation_sequences": len(prepared.validation),
            "test_sequences": len(prepared.test),
        }
    elif args.command == "train-lstm":
        result = sdk.train_world_model()
        payload = {
            "train_losses": result.train_losses,
            "validation_loss": result.validation_loss,
            "checkpoint_path": str(result.checkpoint_path),
        }
    elif args.command == "train-reinforce":
        payload = sdk.train_reinforce()
    elif args.command == "train-a2c":
        payload = sdk.train_a2c()
    else:
        payload = sdk.evaluate_random()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

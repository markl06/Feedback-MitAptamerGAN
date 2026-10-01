import argparse

from mitaptamer.config import Config
from mitaptamer.integrations import read_backend_config
from mitaptamer.training import evaluate, generate, train_evaluator, train_gan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train-evaluator", "train-gan", "evaluate", "generate"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cpu")
        if command.startswith("train-"):
            sub.add_argument("--config")
        if command == "train-evaluator":
            sub.add_argument("--backends", required=True)
        if command in {"train-gan", "generate"}:
            sub.add_argument("--evaluator")
        if command in {"evaluate", "generate"}:
            sub.add_argument("--checkpoint", required=True)
        if command == "evaluate":
            sub.add_argument("--split", choices=["validation", "test"], default="test")
        if command == "generate":
            sub.add_argument("--count", type=int, default=205)
            sub.add_argument("--proposals", type=int, default=10000)
            sub.add_argument("--rounds", type=int, default=10)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.command == "train-evaluator":
        train_evaluator(args.dataset, args.output, Config.load(args.config), read_backend_config(args.backends), args.device)
    elif args.command == "train-gan":
        train_gan(args.dataset, args.output, Config.load(args.config), args.evaluator, args.device)
    elif args.command == "evaluate":
        evaluate(args.dataset, args.checkpoint, args.output, args.device, args.split)
    else:
        generate(args.dataset, args.checkpoint, args.output, args.evaluator, args.count,
                 args.proposals, args.rounds, args.seed, args.device, args.threshold)


if __name__ == "__main__":
    main()

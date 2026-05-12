from __future__ import annotations

import argparse
from pathlib import Path

from yq_ml_alpha.pipelines import materialize, predict, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yq-ml-alpha")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "predict", "run", "materialize"]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "train":
        paths = train.train_only(args.config)
    elif args.command == "predict":
        paths = predict.run(args.config)
    elif args.command == "run":
        paths = train.run(args.config)
    elif args.command == "materialize":
        paths = materialize.run(args.config)
    else:  # pragma: no cover
        raise ValueError(args.command)

    for path in paths:
        print(path)

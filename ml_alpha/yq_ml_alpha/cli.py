from __future__ import annotations

import argparse
from pathlib import Path

from yq_ml_alpha.pipelines import factor, model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yq-ml-alpha")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_commands = [
        "model-train",
        "model-predict",
        "model-run",
        "model-materialize",
        "factor-train",
        "factor-predict",
        "factor-run",
        "factor-materialize",
    ]
    for name in config_commands:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name in {"factor-run", "factor-train"}:
            command.add_argument("--resume", action="store_true", help="resume factor windows from existing outputs/artifacts")
    factor_metadata = subparsers.add_parser("factor-metadata")
    factor_metadata.add_argument("--config", action="append", type=Path, help="refresh one factor config; may be repeated")
    factor_metadata.add_argument("--config-dir", type=Path, help="refresh all *.toml factor configs in this directory")
    args = parser.parse_args(argv)

    if args.command == "model-train":
        paths = model.train_only(args.config)
    elif args.command == "model-predict":
        paths = model.predict_only(args.config)
    elif args.command == "model-run":
        paths = model.run(args.config)
    elif args.command == "model-materialize":
        paths = model.materialize_only(args.config)
    elif args.command == "factor-train":
        paths = factor.train_only(args.config, resume=args.resume)
    elif args.command == "factor-predict":
        paths = factor.predict_only(args.config)
    elif args.command == "factor-run":
        paths = factor.run(args.config, resume=args.resume)
    elif args.command == "factor-materialize":
        paths = factor.materialize_only(args.config)
    elif args.command == "factor-metadata":
        paths = factor.metadata_only(args.config, args.config_dir)
    else:  # pragma: no cover
        raise ValueError(args.command)

    for path in paths:
        print(path)

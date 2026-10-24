from __future__ import annotations

import argparse
import json
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
        if name == "factor-predict":
            command.add_argument("--config", required=True, type=Path, action="append",
                                 help="repeat to predict compatible factors and merge their daily output writes")
        else:
            command.add_argument("--config", required=True, type=Path)
        if name in {"factor-run", "factor-train"}:
            command.add_argument("--resume", action="store_true", help="resume factor windows from existing outputs/artifacts")
        if name in {"model-train", "factor-train"}:
            command.add_argument("--window-id", action="append", help="train only this existing window; repeatable")
            command.add_argument("--list-windows", action="store_true", help="print the training plan without loading samples or models")
            command.add_argument("--aggregate-diagnostics-only", action="store_true", help="merge diagnostics after all window tasks succeed; do not train")
    factor_metadata = subparsers.add_parser("factor-metadata")
    factor_metadata.add_argument("--config", action="append", type=Path, help="refresh one factor config; may be repeated")
    factor_metadata.add_argument("--config-dir", type=Path, help="refresh all *.toml factor configs in this directory")
    factor_metadata_all = subparsers.add_parser("factor-metadata-all")
    factor_metadata_all.add_argument("--config", action="append", type=Path, help="refresh one ml_alpha factor config after Rust metadata; may be repeated")
    factor_metadata_all.add_argument("--config-dir", type=Path, help="refresh all ml_alpha *.toml factor configs in this directory after Rust metadata")
    factor_metadata_all.add_argument("--rust-manifest", type=Path, help="path to factor_engine Cargo.toml")
    args = parser.parse_args(argv)

    if args.command in {"model-train", "factor-train"}:
        if args.aggregate_diagnostics_only and (args.list_windows or args.window_id):
            parser.error("Diagnostics aggregation cannot be combined with window selection/planning")
        if args.list_windows or args.aggregate_diagnostics_only:
            pipeline = factor if args.command == "factor-train" else model
            loader = pipeline._load_factor_config if pipeline is factor else pipeline._load_model_config
            config = loader(args.config)
            if args.list_windows:
                from yq_ml_alpha.pipelines.training_tasks import training_plan
                print(json.dumps(training_plan(config, args.window_id), indent=2))
            else:
                from yq_ml_alpha.pipelines.runtime import _aggregate_diagnostics
                for path in _aggregate_diagnostics(config):
                    print(path)
            return

    if args.command == "model-train":
        paths = model.train_only(args.config, window_ids=args.window_id)
    elif args.command == "model-predict":
        paths = model.predict_only(args.config)
    elif args.command == "model-run":
        paths = model.run(args.config)
    elif args.command == "model-materialize":
        paths = model.materialize_only(args.config)
    elif args.command == "factor-train":
        paths = factor.train_only(args.config, resume=args.resume, window_ids=args.window_id)
    elif args.command == "factor-predict":
        if len(args.config) == 1:
            paths = factor.predict_only(args.config[0])
        else:
            from yq_ml_alpha.pipelines.grouped_predict import predict_many
            paths = predict_many(args.config)
    elif args.command == "factor-run":
        paths = factor.run(args.config, resume=args.resume)
    elif args.command == "factor-materialize":
        paths = factor.materialize_only(args.config)
    elif args.command == "factor-metadata":
        paths = factor.metadata_only(args.config, args.config_dir)
    elif args.command == "factor-metadata-all":
        paths = factor.all_metadata(args.config, args.config_dir, args.rust_manifest)
    else:  # pragma: no cover
        raise ValueError(args.command)

    for path in paths:
        print(path)

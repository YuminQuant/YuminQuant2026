import argparse
import math
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run factor engine in batches by asset/frequency path."
    )
    parser.add_argument("--asset", choices=["chn_stock", "stock", "future"], required=True)
    parser.add_argument("--frequency", choices=["daily", "minute", "minute_1m"], required=True)
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--batch-num",
        type=int,
        default=20,
        help="Number of factors to run in each batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use plan instead of run for every batch.",
    )
    parser.add_argument(
        "--config",
        help="Optional project config.toml path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_num <= 0:
        raise ValueError("--batch-num must be positive")

    factor_engine_dir = Path(__file__).resolve().parents[1]
    manifest_path = factor_engine_dir / "Cargo.toml"
    executable = factor_engine_dir / "target" / "debug" / executable_name()

    subprocess.run(
        ["cargo", "build", "--manifest-path", str(manifest_path)],
        cwd=factor_engine_dir.parent,
        check=True,
    )

    asset_dir = "chn_stock" if args.asset in {"chn_stock", "stock"} else "future"
    cli_asset = "stock" if asset_dir == "chn_stock" else "future"
    freq_dir = "minute" if args.frequency in {"minute", "minute_1m"} else "daily"
    cli_frequency = "minute_1m" if freq_dir == "minute" else "daily"

    factor_path = factor_engine_dir / "src" / "factor" / asset_dir / freq_dir
    file_count = count_factor_files(factor_path)
    factor_ids = list_factor_ids(executable, cli_asset, cli_frequency)

    print(f"factor path: {factor_path}", flush=True)
    print(f"factor .rs files: {file_count}", flush=True)
    print(f"metadata factor ids: {len(factor_ids)}", flush=True)
    if file_count != len(factor_ids):
        print(
            "warning: file count differs from registered factor count; "
            "run metadata and check factor create()/mod.rs exports.",
            file=sys.stderr,
            flush=True,
        )

    batches = chunked(factor_ids, args.batch_num)
    print(f"batch_num: {args.batch_num}", flush=True)
    print(f"batches: {len(batches)}", flush=True)

    command = "plan" if args.dry_run else "run"
    for idx, batch in enumerate(batches, start=1):
        print(f"\n[{idx}/{len(batches)}] {len(batch)} factor(s)", flush=True)
        for factor_id in batch:
            print(f"  {factor_id}", flush=True)

        cmd = [
            str(executable),
            command,
            "--asset",
            cli_asset,
            "--frequency",
            cli_frequency,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--factors",
            ",".join(batch),
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        subprocess.run(cmd, cwd=factor_engine_dir.parent, check=True)


def executable_name():
    return "yq-factor-engine.exe" if sys.platform.startswith("win") else "yq-factor-engine"


def count_factor_files(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"factor path does not exist: {path}")
    return sum(1 for file in path.glob("*.rs") if file.name != "mod.rs")


def list_factor_ids(executable: Path, asset: str, frequency: str):
    completed = subprocess.run(
        [
            str(executable),
            "list",
            "--asset",
            asset,
            "--frequency",
            frequency,
            "--ids-only",
            "true",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def chunked(values, size):
    total = math.ceil(len(values) / size)
    return [values[idx * size : (idx + 1) * size] for idx in range(total)]


if __name__ == "__main__":
    main()

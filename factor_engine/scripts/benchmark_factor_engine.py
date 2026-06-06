import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


PROFILE_RE = re.compile(
    r"date_batch=(?P<date_batch>\d+) factor_batch=(?P<factor_batch>\d+) "
    r"dates=(?P<start_date>\d+)\.\.(?P<end_date>\d+) factors=(?P<factors>\d+) "
    r"load_ms=(?P<load_ms>\d+) compute_ms=(?P<compute_ms>\d+) write_ms=(?P<write_ms>\d+)"
)


SCENARIOS = [
    {
        "name": "stock_daily_demo",
        "asset": "stock",
        "frequency": "daily",
        "start": "20260105",
        "end": "20260130",
        "factors": "pe_zscore_60d,f_momentum_80pec",
    },
    {
        "name": "future_daily_all",
        "asset": "future",
        "frequency": "daily",
        "start": "20260424",
        "end": "20260424",
        "factors": None,
    },
    {
        "name": "future_minute_all",
        "asset": "future",
        "frequency": "minute_1m",
        "start": "20260424",
        "end": "20260424",
        "factors": None,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark factor engine profile output.")
    parser.add_argument("--config", help="Optional project config.toml path.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "reports" / "benchmarks"),
        help="Directory for benchmark CSV/Markdown outputs.",
    )
    parser.add_argument("--threads", help="Optional rayon thread count.")
    return parser.parse_args()


def main():
    args = parse_args()
    factor_engine_dir = Path(__file__).resolve().parents[1]
    repo_root = factor_engine_dir.parent
    manifest_path = factor_engine_dir / "Cargo.toml"
    executable = factor_engine_dir / "target" / "debug" / executable_name()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["cargo", "build", "--manifest-path", str(manifest_path)], cwd=repo_root, check=True)
    metadata_cmd = [str(executable), "metadata"]
    if args.config:
        metadata_cmd.extend(["--config", args.config])
    subprocess.run(metadata_cmd, cwd=repo_root, check=True)

    rows = []
    for scenario in SCENARIOS:
        cmd = [
            str(executable),
            "run",
            "--asset",
            scenario["asset"],
            "--frequency",
            scenario["frequency"],
            "--start-date",
            scenario["start"],
            "--end-date",
            scenario["end"],
            "--profile",
        ]
        if scenario["factors"]:
            cmd.extend(["--factors", scenario["factors"]])
        if args.threads:
            cmd.extend(["--threads", args.threads])
        if args.config:
            cmd.extend(["--config", args.config])

        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(completed.stdout)
        rows.extend(parse_profile_rows(scenario["name"], completed.stdout))

    write_csv(out_dir / "factor_engine_benchmark.csv", rows)
    write_markdown(out_dir / "factor_engine_benchmark.md", rows)
    print(f"wrote {out_dir / 'factor_engine_benchmark.csv'}")
    print(f"wrote {out_dir / 'factor_engine_benchmark.md'}")


def parse_profile_rows(scenario_name, output):
    rows = []
    for line in output.splitlines():
        match = PROFILE_RE.search(line)
        if not match:
            continue
        row = {"scenario": scenario_name}
        row.update({key: int(value) for key, value in match.groupdict().items()})
        row["total_ms"] = row["load_ms"] + row["compute_ms"] + row["write_ms"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    columns = [
        "scenario",
        "date_batch",
        "factor_batch",
        "start_date",
        "end_date",
        "factors",
        "load_ms",
        "compute_ms",
        "write_ms",
        "total_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    columns = ["scenario", "date_batch", "factor_batch", "load_ms", "compute_ms", "write_ms", "total_ms"]
    with path.open("w", encoding="utf-8") as file:
        file.write("# Factor Engine Benchmark\n\n")
        file.write("| " + " | ".join(columns) + " |\n")
        file.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            file.write("| " + " | ".join(str(row[column]) for column in columns) + " |\n")


def executable_name():
    return "yq-factor-engine.exe" if sys.platform.startswith("win") else "yq-factor-engine"


if __name__ == "__main__":
    main()

"""Descriptive annual analyst-forecast error study using local Parquet data."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
# Forecast totals are wan yuan; statement totals are yuan. EPS is yuan/share.
METRICS = {
    "np": ("n_income_attr_p", 10000.0),
    "op_rt": ("revenue", 10000.0),
    "eps": ("basic_eps", 1.0),
}


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(value, format="%Y%m%d", errors="raise")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYYMMDD") from exc


def dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d", errors="coerce",
    )


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise ValueError(f"Cannot read required columns from {path}: {exc}") from exc


def load_forecasts(root: Path, args, audit: dict) -> pd.DataFrame:
    columns = ["ts_code", "report_date", "quarter", "org_name", "author_name",
               "report_title", "create_time", args.metric]
    frames = []
    for year in range(args.start_date.year, args.end_date.year + 1):
        path = root / f"{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing requested forecast year: {path}")
        frames.append(read_columns(path, columns))
    df = pd.concat(frames, ignore_index=True)
    df["report_date"] = dates(df.report_date)
    df = df[df.report_date.between(args.start_date, args.end_date)].copy()
    audit["forecast_rows_in_date_range"] = len(df)
    annual = df.quarter.astype("string").str.fullmatch(r"\d{4}Q4", na=False)
    audit["excluded_non_annual"] = int((~annual).sum())
    df = df[annual].copy()
    df["forecast_year"] = df.quarter.str[:4].astype(int)
    if args.forecast_year:
        df = df[df.forecast_year.isin(args.forecast_year)].copy()
    audit["annual_rows_in_selected_years"] = len(df)
    df["forecast"] = pd.to_numeric(df[args.metric], errors="coerce")
    valid = np.isfinite(df.forecast) & df.ts_code.notna()
    audit["excluded_invalid_forecast_or_code"] = int((~valid).sum())
    df = df[valid].copy()
    df["create_time"] = pd.to_datetime(df.create_time, errors="coerce")
    # One report/author-team/target-year observation; retain earliest stored version.
    # create_time is an update timestamp, NOT a guaranteed first-availability date.
    keys = ["ts_code", "report_date", "org_name", "author_name", "report_title", "quarter"]
    df = df.sort_values(keys + ["create_time"], na_position="last", kind="stable")
    audit["duplicate_report_versions_removed"] = int(df.duplicated(keys).sum())
    df = df.drop_duplicates(keys, keep="first")
    return df.drop(columns=[args.metric]).reset_index(drop=True)


def load_actuals(root: Path, years: set[int], args, audit: dict) -> pd.DataFrame:
    field, divisor = METRICS[args.metric]
    columns = ["ts_code", "end_date", "ann_date", "f_ann_date", "report_type",
               "update_flag", field]
    frames = []
    # Files are partitioned by disclosure year, not necessarily target fiscal year.
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No income statement files: {root}")
    for path in files:
        df = read_columns(path, columns)
        end = dates(df.end_date)
        df = df[(end.dt.month == 12) & (end.dt.day == 31) & end.dt.year.isin(years)].copy()
        if df.empty:
            continue
        df["forecast_year"] = dates(df.end_date).dt.year
        df["actual_date"] = dates(df.f_ann_date).fillna(dates(df.ann_date))
        df = df[df.report_type.isin([1, 4]) & (df.actual_date <= args.as_of)].copy()
        frames.append(df)
    if not frames:
        raise ValueError("No annual actuals for the selected fiscal years/as-of date")
    df = pd.concat(frames, ignore_index=True)
    df["actual"] = pd.to_numeric(df[field], errors="coerce") / divisor
    # Earliest disclosure, type 1 before type 4 on that date, latest flag on ties.
    # A missing first-disclosure metric is NOT replaced by a later restatement.
    df = df.sort_values(["ts_code", "forecast_year", "actual_date", "report_type", "update_flag"],
                        ascending=[True, True, True, True, False], kind="stable")
    tie_keys = ["ts_code", "forecast_year", "actual_date", "report_type", "update_flag"]
    conflicts = df.groupby(tie_keys, dropna=False).actual.nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError("Conflicting financial values with identical version keys; repair source data")
    df = df.drop_duplicates(["ts_code", "forecast_year"], keep="first")
    audit["first_disclosure_actual_records"] = len(df)
    return df[["ts_code", "forecast_year", "actual_date", "actual"]]


def score_forecasts(forecasts, actuals, min_actual, audit):
    df = forecasts.merge(actuals, on=["ts_code", "forecast_year"], how="left", validate="many_to_one")
    valid = np.isfinite(df.actual)
    audit["excluded_missing_or_nonfinite_actual"] = int((~valid).sum())
    df = df[valid].copy()
    valid = df.actual.abs() > min_actual
    audit["excluded_small_actual_denominator"] = int((~valid).sum())
    df = df[valid].copy()
    # Same-day observations are excluded because only dates are known.
    valid = df.report_date < df.actual_date
    audit["excluded_on_or_after_actual_disclosure"] = int((~valid).sum())
    df = df[valid].copy()
    df["signed_error"] = (df.forecast - df.actual) / df.actual.abs()
    df["bias"] = df.signed_error.abs()
    finite = np.isfinite(df.bias)
    audit["excluded_nonfinite_bias"] = int((~finite).sum())
    df = df[finite].copy()
    df["horizon_days"] = (df.actual_date - df.report_date).dt.days
    df["horizon_months"] = ((df.actual_date.dt.year - df.report_date.dt.year) * 12
                            + df.actual_date.dt.month - df.report_date.dt.month)
    audit["scored_forecasts"] = len(df)
    audit["scored_stocks"] = int(df.ts_code.nunique())
    return df.reset_index(drop=True)


def attach_size(df, root: Path, bins: int, max_age: int):
    files = {}
    for path in root.rglob("*.parquet"):
        if len(path.stem) == 8 and path.stem.isdigit():
            day = parse_date(path.stem)
            if day in files:
                raise ValueError(f"Multiple market files for {day}: {path}, {files[day]}")
            files[day] = path
    if not files:
        raise FileNotFoundError(f"No daily YYYYMMDD.parquet market files: {root}")
    days = pd.DatetimeIndex(sorted(files))
    mapping = {}
    for day in df.report_date.unique():
        pos = days.searchsorted(day, side="right") - 1
        if pos >= 0 and (pd.Timestamp(day) - days[pos]).days <= max_age:
            mapping[pd.Timestamp(day)] = days[pos]
    df = df.copy()
    df["size_date"] = df.report_date.map(mapping)
    frames = []
    for day in sorted(set(mapping.values())):
        market = read_columns(files[day], ["ts_code", "trade_date", "total_mv"])
        if not dates(market.trade_date).eq(day).all() or market.ts_code.duplicated().any():
            raise ValueError(f"Invalid date or duplicate stock keys: {files[day]}")
        market["total_mv"] = pd.to_numeric(market.total_mv, errors="coerce")
        market = market[np.isfinite(market.total_mv) & (market.total_mv > 0)].copy()
        # Rank once on the complete market cross section, not forecast rows.
        pct = market.total_mv.rank(method="average", pct=True)
        market["size_group"] = np.ceil(pct * bins).clip(1, bins).astype(int)
        market["size_date"] = day
        frames.append(market[["ts_code", "size_date", "total_mv", "size_group"]])
    if not frames:
        df["total_mv"] = np.nan
        df["size_group"] = np.nan
        return df
    return df.merge(pd.concat(frames, ignore_index=True), on=["ts_code", "size_date"],
                    how="left", validate="many_to_one")


def attach_industry(df, path: Path):
    members = read_columns(path, ["ts_code", "l1_code", "l1_name", "in_date", "out_date"])
    members["in_date"] = dates(members.in_date)
    raw_out = members.out_date
    members["out_date"] = dates(raw_out)
    # Reject malformed nonempty end dates instead of treating them as open-ended.
    malformed = raw_out.notna() & raw_out.astype("string").str.strip().ne("") & members.out_date.isna()
    if malformed.any():
        raise ValueError(f"Malformed industry out_date in {path}")
    members = members[members.in_date.notna() & members.l1_code.notna()].drop_duplicates()
    by_code = {code: group for code, group in members.groupby("ts_code")}
    result = df.copy()
    result["industry_code"] = pd.Series(index=result.index, dtype="string")
    result["industry_name"] = pd.Series(index=result.index, dtype="string")
    # Work on unique stock/date keys. Conflicting active L1 memberships remain missing.
    for code, rows in result.groupby("ts_code"):
        history = by_code.get(code)
        if history is None:
            continue
        for day, indexes in rows.groupby("report_date").groups.items():
            active = history[(history.in_date <= day) & (history.out_date.isna() | (day < history.out_date))]
            if active.l1_code.nunique() == 1:
                row = active.sort_values("in_date").iloc[-1]
                result.loc[indexes, "industry_code"] = row.l1_code
                result.loc[indexes, "industry_name"] = row.l1_name
    return result


def summarize(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return df.groupby(groups, observed=True, dropna=True).agg(
        n=("bias", "size"), stocks=("ts_code", "nunique"),
        mean_bias=("bias", "mean"), median_bias=("bias", "median"),
        p10_bias=("bias", lambda x: x.quantile(.1)),
        p90_bias=("bias", lambda x: x.quantile(.9)),
        mean_signed_error=("signed_error", "mean"),
    ).reset_index()


def make_plot(tables: dict, output: Path, metric: str, min_count: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    settings = [("time", "horizon_months", "Months to first annual disclosure"),
                ("size", "size_group", "Market-cap group (small to large)"),
                ("industry", "industry_code", "Industry (see CSV for names)")]
    for ax, (key, column, label) in zip(axes, settings):
        table = tables[key]
        table = table[table.n >= min_count].copy()
        if key == "industry":
            table = table.sort_values("median_bias", ascending=False)
        if table.empty:
            ax.text(.5, .5, "No groups meet min-count", ha="center", transform=ax.transAxes)
        else:
            ax.bar(np.arange(len(table)), table.median_bias, color="#2875a8")
            ax.set_xticks(np.arange(len(table)), table[column].astype(str), rotation=90 if key == "industry" else 45)
        ax.set_title(label)
        ax.set_ylabel("Median absolute relative error")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{metric}: forecast error by horizon, company size and industry")
    fig.tight_layout()
    fig.savefig(output / "bias_comparison.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, required=True, help="Inclusive report publication date")
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--metric", choices=METRICS, default="np")
    parser.add_argument("--forecast-year", type=int, nargs="+", help="Optional target fiscal year(s)")
    parser.add_argument("--as-of", type=parse_date, default=parse_date(date.today().strftime("%Y%m%d")))
    parser.add_argument("--config", type=Path, default=REPO / "config.toml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--industry", choices=["ci", "sw"], default="ci")
    parser.add_argument("--size-bins", type=int, default=10)
    parser.add_argument("--max-size-age-days", type=int, default=7)
    parser.add_argument("--min-actual-abs", type=float, default=0.0, help="Exclude |actual| <= threshold, in wan yuan or EPS yuan/share")
    parser.add_argument("--min-count", type=int, default=30, help="Minimum observations for a plotted group; CSV retains all groups")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-observations", action="store_true")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date or args.end_date > args.as_of:
        parser.error("Require start-date <= end-date <= as-of")
    if args.size_bins < 2 or args.min_count < 1 or args.max_size_age_days < 0 or not np.isfinite(args.min_actual_abs) or args.min_actual_abs < 0:
        parser.error("Invalid bins, min-count, size age or denominator threshold")
    config = tomllib.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    paths = config.get("paths", {})
    root = args.data_root or Path(paths.get("base_data_dir", REPO / "data"))
    if not root.is_absolute():
        root = REPO / root

    def source(key, default):
        path = Path(paths.get(key, default))
        return path if path.is_absolute() else root / path

    inputs = {
        "forecasts": source("analyst_report_dir", "stock_data/analyst_report"),
        "income": source("fin_income_dir", "stock_data/income"),
        "size": source("stock_daily_basic_dir", "stock_data/daily/basic"),
        "industry": source(f"index_member_{args.industry}_dir", f"index_data/member_{args.industry}") / f"{args.industry}_members.parquet",
    }
    audit = {}
    print("Loading forecasts and first-disclosure actuals...", flush=True)
    forecasts = load_forecasts(inputs["forecasts"], args, audit)
    if forecasts.empty:
        raise ValueError("No forecasts in the selected range/metric/fiscal years")
    actuals = load_actuals(inputs["income"], set(forecasts.forecast_year), args, audit)
    df = score_forecasts(forecasts, actuals, args.min_actual_abs, audit)
    if df.empty:
        raise ValueError(f"No scorable forecasts. Audit: {audit}")
    print(f"Scored {len(df):,} records; attaching historical size and industry...", flush=True)
    df = attach_size(df, inputs["size"], args.size_bins, args.max_size_age_days)
    df = attach_industry(df, inputs["industry"])
    df["size_group"] = df.size_group.astype("Int64")
    audit["missing_size"] = int(df.size_group.isna().sum())
    audit["missing_or_ambiguous_industry"] = int(df.industry_code.isna().sum())
    groups = {"time": ["horizon_months"], "size": ["size_group"], "industry": ["industry_code"]}
    tables = {key: summarize(df, value) for key, value in groups.items()}
    names = df[["industry_code", "industry_name"]].dropna().drop_duplicates("industry_code")
    tables["industry"] = tables["industry"].merge(names, on="industry_code", how="left")
    output = args.output_dir or (Path(__file__).parent / "output" / f"{args.metric}_{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}_{pd.Timestamp.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    for key, table in tables.items():
        table.to_csv(output / f"bias_by_{key}.csv", index=False, encoding="utf-8-sig")
        summarize(df, ["forecast_year"] + groups[key]).to_csv(output / f"bias_by_{key}_and_year.csv", index=False, encoding="utf-8-sig")
    make_plot(tables, output, args.metric, args.min_count)
    if args.save_observations:
        df.to_parquet(output / "observations.parquet", index=False)
    metadata = {"arguments": {key: str(value) if isinstance(value, (Path, pd.Timestamp)) else value for key, value in vars(args).items()},
                "inputs": {key: str(value.resolve()) for key, value in inputs.items()}, "audit": audit,
                "formula": "abs(forecast-actual)/abs(actual); ratio, not percentage points",
                "actual_policy": "first annual disclosure, types 1/4; same-date type 1 first; no restatement fallback",
                "time_policy": "publication to realized first disclosure; ex-post descriptive study, not a backtest signal",
                "industry_policy": "historical [in_date, out_date); ambiguous memberships excluded only from industry summaries",
                "weighting": "equal weight per distinct report/author-team/target fiscal year; no winsorization"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

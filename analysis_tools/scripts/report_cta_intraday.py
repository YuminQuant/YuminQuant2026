"""Generate a single-product CTA intraday diagnostic report."""
from datetime import datetime
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis_tools"
sys.path.insert(0, str(ANALYSIS))

from yq_analysis.cta_intraday import load_intraday_report


def _date(value):
    try:
        return int(datetime.strptime(value.replace("-", ""), "%Y%m%d").strftime("%Y%m%d"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYYMMDD or YYYY-MM-DD") from exc


def generate_report(
    signal, product, start, end, *, root, market_root, signal_root, mapping_root, hdfs_host, output, calendar_path=None,
):
    from plotly.offline import get_plotlyjs

    report = load_intraday_report(signal, product, start, end, root=root, market_root=market_root,
                                  signal_root=signal_root, mapping_root=mapping_root,
                                  calendar_path=calendar_path, hdfs_host=hdfs_host)
    template = (ANALYSIS / "templates/cta_intraday.html").read_text(encoding="utf-8")
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = template.replace("__PLOTLY_JS__", get_plotlyjs()).replace("__DASHBOARD_DATA__", embedded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", required=True, help="Backtest directory and prediction column name")
    parser.add_argument("--product", required=True, type=str.upper, help="Futures product code, e.g. AG")
    parser.add_argument("--start-date", required=True, type=_date)
    parser.add_argument("--end-date", required=True, type=_date)
    parser.add_argument("--root", type=Path, default=ROOT / "data/strategy/future")
    parser.add_argument("--market-root", type=Path, default=ROOT / "data/future_data/minute")
    parser.add_argument("--signal-root", type=Path, default=ROOT / "data/models/future/minute_1m")
    parser.add_argument("--mapping-root", type=Path, default=ROOT / "data/future_data/basic/mapping")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/calendar/trade_cal_SHFE.parquet",
                        help="Futures trading calendar used for the threshold lookback")
    parser.add_argument("--hdfs-host", default="xz336", help="SSH host for missing local predictions and main contracts")
    parser.add_argument("--local-only", action="store_true", help="Read only local input files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.signal):
        parser.error("--signal must be a single backtest directory name")
    if not re.fullmatch(r"[A-Z]+", args.product):
        parser.error("--product must be a futures product code without an exchange suffix")
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    default_name = f"{args.signal}_{args.product}_{args.start_date}_{args.end_date}_intraday.html"
    output = args.output or ROOT / "data/reports/cta" / default_name
    result = generate_report(args.signal, args.product, args.start_date, args.end_date, root=args.root,
                             market_root=args.market_root, signal_root=args.signal_root, mapping_root=args.mapping_root,
                             hdfs_host=None if args.local_only else args.hdfs_host, output=output,
                             calendar_path=args.calendar)
    print(result.resolve())


if __name__ == "__main__":
    main()

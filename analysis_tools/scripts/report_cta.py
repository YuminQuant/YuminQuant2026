"""Generate a standalone HTML report from local CTA backtest outputs."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis_tools"
sys.path.insert(0, str(ANALYSIS))

from yq_analysis.cta_report import load_cta_report_views
from yq_analysis.future_io import load_future_sectors


def _rows(table):
    return json.loads(table.reset_index().to_json(orient="records", double_precision=12))


def _product_labels(products, sectors):
    lookup = sectors.set_index("product")
    labels = {}
    for code in products:
        if code in lookup.index:
            row = lookup.loc[code]
            labels[code] = {"name": row.chinese, "sector": str(row.abbr), "sector_name": row["name"]}
        else:
            labels[code] = {"name": "", "sector": "Other", "sector_name": "未分类"}
    return labels


def _encode_pnl(curves):
    data = {}
    for code, series in curves.items():
        start = int(curves.index.get_loc(series.first_valid_index()))
        data[code] = {"start": start, "values": series.iloc[start:].round(6).tolist()}
    return data


def generate_report(signal, *, root, market_root, output):
    sectors = load_future_sectors(ROOT / "factor_engine/docs/FUTURE_SECTORS.md")
    scopes = {"product:" + code: [code] for code in sectors['product']}
    scopes.update({"sector:" + str(abbr): group['product'].tolist()
                   for abbr, group in sectors.groupby('abbr')})
    portfolio, product_tables, curves, scoped, active_dates = load_cta_report_views(
        signal, root=root, market_root=market_root, product_scopes=scopes)
    for code in curves.columns:
        if "product:" + code not in scoped:
            raise ValueError(f"Missing portfolio scope for {code}; update FUTURE_SECTORS.md")
    data = {
        "signal": signal,
        "dates": [f"{date // 10000:04d}-{date // 100 % 100:02d}-{date % 100:02d}" for date in curves.index],
        "products": _product_labels(curves.columns, sectors),
        "portfolio": _rows(portfolio),
        "portfolio_scopes": {scope: _rows(table) for scope, table in scoped.items()},
        "product_tables": {str(period): _rows(table) for period, table in product_tables.items()},
        "product_columns": list(product_tables["All"].reset_index().columns),
        "portfolio_columns": list(portfolio.reset_index().columns),
        "pnl": _encode_pnl(curves),
        "active_dates": active_dates,
    }
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (ANALYSIS / "templates/cta_report.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__CTA_DATA__", embedded), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", required=True, help="CTA backtest directory name, e.g. mdl_100000")
    parser.add_argument("--root", type=Path, default=ROOT / "data/strategy/future", help="Root of local strategy outputs")
    parser.add_argument("--market-root", type=Path, help="Override the local futures minute data root")
    parser.add_argument("--output", type=Path, help="Override the generated HTML path")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.signal):
        parser.error("--signal must be a single backtest directory name")
    output = args.output or ROOT / "data/reports/cta" / f"{args.signal}.html"
    print(generate_report(args.signal, root=args.root, market_root=args.market_root, output=output).resolve())


if __name__ == "__main__":
    main()

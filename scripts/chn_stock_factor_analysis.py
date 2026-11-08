"""Incrementally update the existing stock factor review from local backtests."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "data/reports/factor_review_20160101.html"
DEFAULT_PLOTS = ROOT / "analysis_tools/plots/factor_review_2016_2026"
sys.path.insert(0, str(ROOT / "analysis_tools"))
from yq_analysis.io import load_backtest_result
from yq_analysis.metrics import annual_return
from yq_analysis.report import make_backtest_report, make_ic_report, make_return_report, make_return_report_by_year

START = 20160101
PER_YEAR = 240
ANN = "annual_return(%)"
GROUPS = [f"group_{n}" for n in range(1, 11)]
FILES = ("returns", "ic", "factor_stats", "barra_exposure", "index_group_returns")
WINDOWS = [("2016\u20132020", 20160101, 20201231), ("2021\u20132026", 20210101, 20261231)]
OK, ADJ, BAD, NA = "达标", "调整后达标", "未达标", "指标不全"
PAYLOAD = re.compile(r'<script type="application/json" id="report-data">([\s\S]*?)</script>')
ID = re.compile(r"[A-Za-z0-9_-]+")


def fingerprint(folder):
    files = {}
    for name in FILES:
        path = folder / f"{name}.parquet"
        if path.is_file():
            stat = path.stat()
            files[name] = [stat.st_size, stat.st_mtime_ns]
    return files


def digest(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def finite(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def get(frame, index, key):
    return finite(frame.loc[index, key]) if index in frame.index and key in frame else None


def perf(frame, key="return"):
    return make_return_report(frame, return_col=key, periods_per_year=PER_YEAR).set_index("portfolio")


def ann(values):
    return annual_return(values, periods_per_year=PER_YEAR) * 100


def endpoints(a, e):
    return {f"g{n}_{suffix}": get(frame, f"group_{n}", ANN)
            for n in (1, 10) for suffix, frame in (("abs", a), ("ex", e))}


def dated(frame, key="trade_date"):
    return frame[frame[key].ge(START)].copy()


def plot_file(ctx, id, mode):
    relative = Path(mode) / f"{id}.jpg"
    return ctx["staged"] / relative, Path(os.path.relpath(ctx["plots"] / relative, ctx["report"].parent)).as_posix()


def publish_report(ctx, new_html, obsolete):
    """Publish only changed figures, restoring them if the HTML cannot be replaced."""
    staged = ctx["staged"]
    replacements = {ctx["plots"] / p.relative_to(staged): p for p in staged.glob("*/*.jpg")}
    actions = {**dict.fromkeys(obsolete), **replacements}
    backup = staged.parent / "previous_plots"
    applied = []
    try:
        for index, (target, source) in enumerate(actions.items()):
            if source is None and not target.is_file():
                continue
            prior = None
            if target.exists():
                backup.mkdir(exist_ok=True)
                prior = backup / f"{index}.jpg"
                shutil.copy2(target, prior)
            if source is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            applied.append((target, prior))
        os.replace(new_html, ctx["report"])
    except OSError:
        for target, prior in reversed(applied):
            if prior is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(prior, target)
        raise


def folded(id, returns, screen, sha, ctx):
    from matplotlib import pyplot as plt

    source = returns[returns.portfolio.isin(GROUPS)]
    a, e, b = [source.pivot(index="trade_date", columns="portfolio", values=k).reindex(columns=GROUPS)
               for k in ("return", "excess_return", "benchmark_return")]
    valid = np.isfinite(a).all(axis=1) & np.isfinite(e).all(axis=1) & np.isfinite(b).all(axis=1)
    a, e, b = a[valid], e[valid], b[valid]
    if not np.allclose(b.to_numpy(), b.iloc[:, 0].to_numpy()[:, None]):
        raise ValueError(f"{id}: original groups do not share a benchmark")
    if a.empty:
        return dict(error="原始10组没有共同有效收益日期", source_sha256=sha)
    frames = [pd.DataFrame({"trade_date": a.index, "portfolio": f"group_{k+1}",
                           "return": (.5 * (a.iloc[:, k] + a.iloc[:, 9-k])).to_numpy(),
                           "excess_return": (.5 * (e.iloc[:, k] + e.iloc[:, 9-k])).to_numpy()}) for k in range(5)]
    data = pd.concat(frames, ignore_index=True)
    aa, ee = perf(data), perf(data, "excess_return")
    ya, ye = [make_return_report_by_year(data, return_col=k, periods_per_year=PER_YEAR)
              .set_index(["year", "portfolio"]) for k in ("return", "excess_return")]
    years = [dict(year=int(y), portfolio=g, absolute_ann=get(ya, (y, g), ANN), excess_ann=get(ye, (y, g), ANN),
                  sharpe=get(ya, (y, g), "sharpe"), excess_ir=get(ye, (y, g), "sharpe"),
                  max_drawdown=get(ya, (y, g), "max_drawdown(%)")) for y, g in ya.index]
    periods = []
    for period, lo, hi in WINDOWS:
        x = data[data.trade_date.between(lo, hi)]
        pa, pe = perf(x), perf(x, "excess_return")
        periods.extend(dict(period=period, portfolio=g, absolute_ann=get(pa, g, ANN), excess_ann=get(pe, g, ANN))
                       for g in pa.index)
    rank = pd.Series([ann(f.excess_return) for f in frames]).rank()
    path, link = plot_file(ctx, id, "folded")
    out = dict(absolute=records(aa.reset_index()), excess=records(ee.reset_index()), by_year=years, periods=periods,
               n=len(a), excluded=int((~valid).sum()), start=int(a.index[0]), end=int(a.index[-1]),
               shape=finite(pd.Series(range(1, 6)).corr(rank)) if rank.nunique() > 1 else None,
               source_sha256=sha, plot=link, outer_absolute=get(aa, "group_1", ANN), center_absolute=get(aa, "group_5", ANN),
               outer_excess=get(ee, "group_1", ANN), center_excess=get(ee, "group_5", ANN))
    if screen:
        ls = round(ann(.5 * (frames[-1]["return"] - frames[0]["return"])), 2)
        best = max(out["outer_excess"], out["center_excess"])
        out["screening"] = dict(ls_ann=ls, best_ex=best, outer_excess=out["outer_excess"],
                                center_excess=out["center_excess"], n=out["n"], start=out["start"], end=out["end"],
                                long_group="F5", short_group="F1", gross_exposure=1.0, passed=ls > 0 and best > 0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), layout="constrained")
    try:
        for ax, absr, exr, n, prefix in (
            (axes[0, 0], [ann(a[g]) for g in GROUPS], [ann(e[g]) for g in GROUPS], 10, "G"),
            (axes[0, 1], aa[ANN], ee[ANN], 5, "F"),
        ):
            x = np.arange(n)
            ax.bar(x-.18, absr, .36, label="Absolute", color="#397a60")
            ax.bar(x+.18, exr, .36, label="Excess", color="#b48922")
            ax.set_xticks(x, [f"{prefix}{j+1}" for j in x])
            ax.set_title(f"{n} groups | Annual return (%)")
            ax.legend(frameon=False)
        for k, frame in enumerate(frames, 1):
            dates = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
            for ax, key in zip(axes[1], ("return", "excess_return")):
                ax.plot(dates, ((1 + frame[key]).cumprod()-1)*100, label=f"F{k}: G{k}+G{11-k}", lw=1)
        for ax, title in zip(axes[1], ("Cumulative absolute return (%)", "Cumulative excess return (%)")):
            ax.set_title(title)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(id + " | Folded group approximation")
        fig.savefig(path, dpi=105)
    finally:
        plt.close(fig)
    return out


def settle(root, id, old, detail, category, ctx):
    from matplotlib import pyplot as plt
    from yq_analysis.plots import plot_return_summary

    tables = load_backtest_result(root, id)
    r, i, s, b, ix = [tables[k] for k in FILES]
    for key, frame in tables.items():
        if frame is None or "factor_id" not in frame or set(frame.factor_id.dropna()) - {id}:
            raise ValueError(f"{id}: missing or mismatched {key}.parquet")
    r, s, ix = [dated(x) for x in (r, s, ix)]
    i = dated(i, "factor_date")
    main = i[i.horizon.eq(1)]
    b = b[b.trade_date.ge(START) | b.trade_date.isna()]
    if not set(GROUPS + ["long_short"]) <= set(r.portfolio) or r.duplicated(["trade_date", "portfolio"]).any():
        raise ValueError(f"{id}: expected unique daily portfolios for ten groups and long_short")
    report = make_backtest_report(r, main, s, periods_per_year=PER_YEAR)
    a, e = [report[k].set_index("portfolio") for k in ("portfolio_total", "excess_total")]
    ic = report["ic"].set_index("metric")
    q = endpoints(a, e)
    best, group = max([(q[f"g{n}_ex"], f"group_{n}") for n in (1, 10) if q[f"g{n}_ex"] is not None],
                      default=(None, None), key=lambda item: item[0])
    ls = a.loc["long_short"]
    ls_ann, n = finite(ls[ANN]), int(ls.observations)
    cov = finite(s.coverage.mean()*100)
    issues = []
    if cov is None:
        issues.append("无有效覆盖率")
    elif cov < 50:
        issues.append("平均覆盖率不足50%")
    if n < 480:
        issues.append("有效多空日收益不足480日")
    signs, selected = b.rank_ic_sign.dropna().unique(), b.selected_group.dropna().unique()
    if len(signs) > 1 or len(selected) > 1:
        raise ValueError(f"{id}: inconsistent saved RankIC direction")
    sign, selected = (int(signs[0]) if len(signs) else None), (selected[0] if len(selected) else None)
    status = NA if ls_ann is None or best is None else OK if ls_ann > 0 and best > 0 else BAD
    f = dict(id=id, category=category, state="活跃", tags=old.get("tags", []), original_status=status, status=status,
             ls_ann=ls_ann, ls_sharpe=finite(ls["sharpe"]), ls_mdd=finite(ls["max_drawdown(%)"]),
             ls_vol=finite(ls["annual_volatility(%)"]), ls_cum=finite(ls["cumulative_return(%)"]),
             ls_win=finite(ls["win_rate(%)"]), best_ex=best, best_group=group, ic=get(ic, "IC", "mean"),
             ic_ir=get(ic, "IC", "ir"), rank_ic=get(ic, "RankIC", "mean"), rank_ir=get(ic, "RankIC", "ir"),
             ic_n=int(get(ic, "IC", "observations") or 0), n=n, coverage=cov, zero_days=int(s.coverage.eq(0).sum()),
             start=int(r.trade_date.min()), end=int(r.trade_date.max()), recent_ann=get(perf(r[r.trade_date>=20240101]), "long_short", ANN),
             direction_sign=sign, direction_group=selected, direction_source="saved_backtest_direction",
             oriented_excess_ann=get(e, selected, ANN), oriented_excess_ir=get(e, selected, "sharpe"), issues=issues, **q)
    yearly = []
    for year in sorted((r.trade_date//10000).unique()):
        aa, ee = [report[k].loc[report[k].year.eq(year)].set_index("portfolio") for k in ("portfolio_by_year", "excess_by_year")]
        ii = make_ic_report(main[main.factor_date//10000==year]).set_index("metric")
        yearly.append(dict(year=int(year), ls_ann=get(aa, "long_short", ANN), ls_mdd=get(aa, "long_short", "max_drawdown(%)"),
                           ic=get(ii, "IC", "mean"), rank_ic=get(ii, "RankIC", "mean"),
                           coverage=finite(s.loc[s.trade_date//10000==year, "coverage"].mean()*100), **endpoints(aa, ee)))
    complete = [row for row in yearly if row["year"] < 2026 and row["ls_ann"] is not None]
    f.update(year_n=len(complete), year_pos=sum(row["ls_ann"] > 0 for row in complete))
    split = []
    for period, lo, hi in WINDOWS:
        x = r[r.trade_date.between(lo, hi)]
        ee = perf(x, "excess_return")
        split.append(dict(period=period, ls_ann=get(perf(x), "long_short", ANN),
                          g1_ex=get(ee, "group_1", ANN), g10_ex=get(ee, "group_10", ANN)))
    sha = digest(root / id / "returns.parquet")
    fold = folded(id, r, status==BAD, sha, ctx)
    if screen := fold.get("screening"):
        f.update(adjusted_ls_ann=screen["ls_ann"], adjusted_best_ex=screen["best_ex"])
        if screen["passed"]:
            f["status"] = ADJ
    path, link = plot_file(ctx, id, "original")
    fig = plot_return_summary(r, groups=10, barra_exposure=b, index_group_returns=ix, ic=i,
                              save=False, figsize=(19, 9), title=id)
    try:
        fig.savefig(path, dpi=105, bbox_inches="tight")
    finally:
        plt.close(fig)
    d = dict(name=detail.get("name", id), description=detail.get("description", ""), version=detail.get("version", ""),
             returns=records(report["portfolio_total"]), excess=records(report["excess_total"]), ic=records(report["ic"]),
             stats=records(report["factor_stats"]), yearly=yearly, split=split, issues=issues, plot=link, folded=fold,
             barra=records(b.groupby(["metric", "barra_factor"], as_index=False).value.mean()))
    return f, d


def median(rows, key, absolute=False):
    values = [abs(f[key]) if absolute else f[key] for f in rows if f[key] is not None]
    return float(np.median(values)) if values else None


def summarize(data):
    factors = data["factors"]
    c = Counter(f["status"] for f in factors)
    if not factors:
        raise ValueError("Cannot publish an empty report")
    data["summary"].update(total=len(factors), passed=c[OK]+c[ADJ], failed=c[BAD], unavailable=c[NA],
        original_passed=c[OK], adjusted_passed=c[ADJ], original_failed=sum(f["original_status"]==BAD for f in factors),
        low_cov=sum(f["coverage"] is not None and f["coverage"]<50 for f in factors), no_valid=sum(f["n"]==0 for f in factors),
        short=sum(f["n"]<480 for f in factors), positive_ls=sum((f["ls_ann"] or 0)>0 for f in factors),
        positive_ex=sum((f["best_ex"] or 0)>0 for f in factors), median_ls=median(factors,"ls_ann"),
        median_ex=median(factors,"best_ex"), median_cov=median(factors,"coverage"),
        unknown_meta=sum(f["state"]=="未匹配元数据" for f in factors),
        first=min((f["start"] for f in factors if f["start"]), default=None),
        last=max((f["end"] for f in factors if f["end"]), default=None))
    categories = []
    for cat in sorted({f["category"] for f in factors}):
        part = [f for f in factors if f["category"]==cat]
        c = Counter(f["status"] for f in part)
        categories.append(dict(category=cat,count=len(part),passed=c[OK]+c[ADJ],original_passed=c[OK],
                               adjusted_passed=c[ADJ],failed=c[BAD],ls_median=median(part,"ls_ann"),
                               ex_median=median(part,"best_ex"),rank_median=median(part,"rank_ic",True),
                               coverage_median=median(part,"coverage"),
                               low_cov=sum(f["coverage"] is not None and f["coverage"]<50 for f in part)))
    data["categories"] = categories
    data["preferred"] = [f["id"] for f in sorted((f for f in factors if f["status"]==OK and (f["coverage"] or 0)>=80
        and f["n"]>=480 and f["ls_sharpe"] is not None), key=lambda f:-f["ls_sharpe"])[:12]]
    data["weak"] = [f["id"] for f in sorted((f for f in factors if f["status"]==BAD),key=lambda f:f["ls_ann"] or 0)[:10]]
    data["risk"] = [f["id"] for f in sorted((f for f in factors if (f["coverage"] or 0)<50 or f["n"]<480),
                                           key=lambda f:f["coverage"] or 0)]


def correlations(data, root, changed):
    details = data["details"]
    method = data["correlation_method"]
    series = {}
    for id, detail in details.items():
        path = root / id / "returns.parquet"
        sha = detail["folded"].get("source_sha256")
        if sha and path.is_file() and digest(path)==sha:
            rows = pd.read_parquet(path, columns=["trade_date","portfolio","return"],
                                   filters=[("portfolio","=","long_short"), ("trade_date",">=",START)])
            series[id] = rows.set_index("trade_date")["return"]
    verified = set(series)
    # The original HTML stores only the top pairs. Retain known pairs when one
    # input is unavailable, but never carry forward a pair with a replaced leg.
    known = {}
    for p in data.get("known_correlations", data["correlations"]):
        legs = {p["left"], p["right"]}
        if legs <= details.keys() and not legs & changed and not legs <= verified:
            known[tuple(sorted(legs))] = p
    if series:
        panel = pd.DataFrame(series).sort_index().reindex(columns=sorted(verified)).replace([np.inf,-np.inf],np.nan)
        values = panel.corr(min_periods=method["min_common_days"])
        mask = panel.notna().to_numpy(dtype=np.int32)
        days = mask.T @ mask
        for j, left in enumerate(panel.columns):
            for k in range(j+1, len(panel.columns)):
                v = values.iat[j,k]
                if v >= method["threshold"]:
                    right = panel.columns[k]
                    known[left,right] = dict(left=left,right=right,correlation=round(float(v),4),days=int(days[j,k]))
    pairs = sorted(known.values(), key=lambda p:(-p["correlation"], p["left"], p["right"]))
    data["known_correlations"] = pairs
    data["correlations"] = pairs[:method["top_n"]]
    data["summary"]["corr_count"] = len(pairs)
    data["added_correlations"] = [p for p in pairs if {p["left"], p["right"]} & set(data["added"])][:method["top_n"]]
    data["correlation_input"] = dict(verified=len(verified), total=len(details))


def overview(data):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    try:
        for status, label, color in [(OK,"Passed","#397a60"), (ADJ,"Folded pass","#b48922"),
                                     (BAD,"Not passed","#b15853"), (NA,"Unavailable","#7a7a7a")]:
            rows = [f for f in data["factors"] if f["status"]==status and f["ls_ann"] is not None and f["coverage"] is not None]
            if rows:
                axes[0].scatter([f["coverage"] for f in rows], [f["ls_ann"] for f in rows], label=label, color=color, s=12)
        if axes[0].collections:
            axes[0].legend(frameon=False)
        axes[0].set(xlabel="Coverage (%)", ylabel="Annual long-short return (%)")
        axes[1].hist([f["ls_ann"] for f in data["factors"] if f["ls_ann"] is not None], bins=26, color="#397a60")
        if data["summary"]["median_ls"] is not None:
            axes[1].axvline(data["summary"]["median_ls"], linestyle="--", color="#b48922")
        axes[1].set(xlabel="Annual long-short return (%)", ylabel="Factors")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backtest-root", type=Path, default=ROOT/"data/backtest/stock/daily")
    parser.add_argument("--plots-root", type=Path, help="Exact directory for the report's original/ and folded/ figures.")
    parser.add_argument("--remove", nargs="+", default=[], metavar="ID", help="Remove from HTML only; remember the exclusion.")
    parser.add_argument("--refresh", nargs="+", default=[], metavar="ID", help="Replace or restore selected IDs from local results.")
    parser.add_argument("--category", action="append", default=[], metavar="ID=LABEL", help="Set a display category.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report, root = args.report.resolve(), args.backtest_root.resolve()
    plot_root = (args.plots_root or (DEFAULT_PLOTS if report == DEFAULT_REPORT.resolve()
                 else report.parent / f"{report.stem}_plots")).resolve()
    if not root.is_dir():
        parser.error(f"Backtest directory not found: {root}")
    text = report.read_text(encoding="utf-8")
    match = PAYLOAD.search(text)
    if not match:
        parser.error("Expected an existing factor review HTML containing report-data.")
    data = json.loads(match[1])
    old = {f["id"]:f for f in data["factors"]}
    state = data.get("local_refresh", {})
    if state.get("root", str(root)) != str(root):
        parser.error("Backtest root differs from the root previously registered in this report.")
    prev = state.get("files", {})
    folders = {p.name:p for p in root.iterdir() if p.is_dir() and ID.fullmatch(p.name)}
    current = {id:fingerprint(p) for id,p in folders.items()}
    complete = {id for id, files in current.items() if set(files)==set(FILES)}
    remove, refresh = set(args.remove), set(args.refresh)
    if remove & refresh or not refresh <= complete or not all(ID.fullmatch(x) for x in remove|refresh):
        parser.error("Removal and refresh must be disjoint; refresh requires all five result files.")
    removed = set(old) & (remove | (prev.keys() - folders.keys()))
    excluded = (set(state.get("excluded", [])) | remove) - refresh
    added = complete - old.keys() - excluded
    changed = added | refresh | {id for id in old.keys() & complete & prev.keys()
                                 if id not in excluded and current[id]!=prev[id]}
    categories = {}
    for assignment in args.category:
        id, sep, label = assignment.partition("=")
        if not sep or not label.strip() or id not in ((old.keys() - removed) | changed):
            parser.error(f"Invalid category assignment: {assignment}")
        categories[id] = label.strip()
    print("Add:", ", ".join(sorted(added)) or "-")
    print("Refresh:", ", ".join(sorted(changed-added)) or "-")
    print("Remove:", ", ".join(sorted(removed)) or "-")
    incomplete = {id for id, files in current.items() if files and id not in complete and id not in excluded}
    if incomplete:
        print("Skipped incomplete results:", ", ".join(sorted(incomplete)), file=sys.stderr)
    if args.dry_run:
        return
    current_state = dict(root=str(root), files={id:current[id] if id in complete else prev[id]
                         for id in (complete | (prev.keys() & folders.keys())) if id not in excluded},
                         excluded=sorted(excluded))
    if not (changed or removed or categories) and state==current_state:
        print("No changes.")
        return
    data["local_refresh"] = current_state
    old_plots = {}
    for id in changed | removed:
        detail = data["details"].get(id, {})
        for mode, part in (("original", detail), ("folded", detail.get("folded", {}))):
            expected = plot_root / mode / f"{id}.jpg"
            if part.get("plot") and (report.parent / part["plot"]).resolve() == expected:
                old_plots[id, mode] = expected
    for id in removed:
        del old[id]
        del data["details"][id]
    # The first run registers the local baseline without replacing old metrics.
    substantive = bool(changed or removed or categories)
    tmp = ROOT/"tmp"
    tmp.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="stock-factor-review-", dir=tmp) as directory:
        stage = Path(directory)
        os.environ.setdefault("MPLCONFIGDIR", str(stage/"mpl"))
        ctx = dict(staged=stage/"plots", plots=plot_root, report=report)
        if substantive:
            import matplotlib
            matplotlib.use("Agg")
            for mode in ("original", "folded"):
                (ctx["staged"]/mode).mkdir(parents=True, exist_ok=True)
            for id in sorted(changed):
                item = old.get(id, {})
                fallback = "端到端" if (ROOT/"ml_alpha/factors"/f"{id}.toml").is_file() else "未分类"
                category = categories.get(id, item.get("category", fallback))
                f, d = settle(root,id,item,data["details"].get(id,{}),category,ctx)
                if fingerprint(folders[id])!=current[id]:
                    raise RuntimeError(f"{id}: backtest files changed while reading")
                old[id], data["details"][id] = f, d
            for id, category in categories.items():
                old[id]["category"] = category
            data["factors"] = sorted(old.values(), key=lambda f:f["id"])
            data["added"] = sorted(added) if changed else [id for id in data["added"] if id in old]
            summarize(data)
            correlations(data,root,changed)
            data["generated"] = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
            data["source"] = data["source"].split(" | 本地增量",1)[0] + " | 本地增量：使用本地回测产物，未重新回测"
        payload = json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
        out = text[:match.start(1)]+payload+text[match.end(1):]
        if substantive:
            chart = r'(src="data:image/png;base64,)[^"]+(" alt="全库因子覆盖率与多空年化收益散点图，以及多空年化收益分布直方图")'
            out, count = re.subn(chart,lambda m:m[1]+overview(data)+m[2],out)
            if count!=1:
                raise ValueError("Overview chart not found in the report template")
            if 'id="local-correlation-scope"' not in out:
                note = '<p id="local-correlation-scope" class="small muted"></p>'
                out = out.replace('<div id="correlation-table">',note+'<div id="correlation-table">',1)
                js = '''const lc=D.correlation_input;
if(lc && lc.verified<lc.total) document.getElementById('local-correlation-scope').textContent=
 '本地收益与报告版本一致：'+lc.verified+'/'+lc.total+' 个因子。仅重算可核对的因子对，其余沿用仍有效的历史已知结果；高相关对数是已知下限。';
'''
                out = out.replace("window.__reportQA=",js+"window.__reportQA=",1)
            out = out.replace("发现${S.corr_count}对因子","报告记录${S.corr_count}对因子")
            out = out.replace("类别按回测快照元数据标签做互斥归类；带 e2e 或 model_generated 的因子归入“端到端”，其他混合因子保留“分析师+基本面/量价”。",
                              "已有因子沿用原分类；新增因子默认未分类，项目端到端因子归入“端到端”，可在更新时指定分类。")
        assert set(data["details"])=={f["id"] for f in data["factors"]}
        assert json.loads(PAYLOAD.search(out)[1])==data
        if report.read_text(encoding="utf-8")!=text:
            raise RuntimeError("Report changed while updating; run the command again")
        new_html = stage/"report.html"
        new_html.write_text(out,encoding="utf-8")
        obsolete = [path for (id, mode), path in old_plots.items()
                    if id in removed or (mode == "folded" and not data["details"][id]["folded"].get("plot"))]
        publish_report(ctx, new_html, obsolete)
    print(f"Updated {report} ({len(data['factors'])} factors)")


if __name__ == "__main__":
    main()

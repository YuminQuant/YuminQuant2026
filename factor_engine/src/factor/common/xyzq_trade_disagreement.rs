use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::stock_minute_raw_spec;
use crate::factor::Factor;
use crate::operators::ts_mean;

#[path = "xyzq_trade_disagreement_math.rs"]
mod math;
use math::{
    basic_onset, bayes_onset, corr, csad_fit, mean, onset, prior, quantile_residual, sparse_gamma,
    Counts,
};

const VERSION: &str = "0.1.0";
const IDS: [&str; 10] = [
    "vr_quantile_resid_corr",
    "vr_q80_ret_corr",
    "lr_score_mor",
    "lr_beta2_aft_minus_mor",
    "sr_dispersion_mean_vwap",
    "sr_dispersion_volume_absret_corr",
    "high_vol_onset",
    "low_vol_onset",
    "bayes_high_vol_onset",
    "bayes_low_vol_onset",
];
const ALIASES: [&str; 10] = [
    "vr_quantile_resid_corr",
    "vr_q80_ret_corr",
    "lr_score_mor",
    "lr_beta2_aft_minus_mor",
    "sr_dispersion_mean_vwap",
    "sr_dispersion_volume_absret_corr",
    "highVol_onset",
    "lowVol_onset",
    "bayesHighVol_onset",
    "bayesLowVol_onset",
];

fn raw_id(i: usize) -> String {
    format!("{}_raw", IDS[i])
}
fn raw_spec(i: usize) -> IntradayDailyRawSpec {
    let columns: &[&str] = match i {
        2 | 3 => &["open", "close"],
        4 => &["open", "close", "vol", "amount"],
        _ => &["open", "close", "vol"],
    };
    stock_minute_raw_spec(&raw_id(i), VERSION, columns, 1)
}

pub struct TradeDisagreement(pub usize);
impl Factor for TradeDisagreement {
    fn spec(&self) -> FactorSpec {
        let i = self.0;
        FactorSpec {
            id: IDS[i].into(), aliases: vec![ALIASES[i].into()], name: ALIASES[i].into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: VERSION.into(),
            tags: ["XYZQ", "price_volume", "intraday", "minute_agg", "volume", "daily", "neutralize", "barra", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: format!("{}: single-stock/day minute models, 09:31-11:30 and 13:01-15:00. First return close/open-1, later adjacent close/close-1; do not bridge missing slots. Equal-weight non-BJ market (no extra ST/limit filter). Quantile loss at .2/.8 on intercept, signed return, previous relative volume, first/last 30 slot indicators; >=20 valid rows, skip lag-missing rows. CSAD OLS on intercept, abs(market),100*market^2; >=4 rows and full rank. Sparse no-intercept slope plus diagonal market^2 responses: support exchanges to a one-swap local optimum, s=min(24,floor(valid_minutes/10)), >=10 rows; approximation, not an author-disclosed s. Pearson correlations, population statistics, invalid/constant regressions null. Onset gamma<=0 and next five scheduled trading minutes all valid, no past-five filter, no overlap deduplication; crossing quantile predictions excluded only from onset. Log(control/event); basic counts >=1/1, Bayes >=3/5. Separate market priors per group: smoothed m, n-weighted variance about m, noise=m*(1-m)*M/sum(n), between>=1e-8, kappa in [5,200]. {} Then SW L1+SIZE neutralization, no rank/zscore/winsorization/fill/sign flip. Excludes BJ; no trade-count proxy.", ALIASES[i], if i<6 {"10 trading-day raw mean, min_periods=1."} else {"No rolling smoothing."}),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily,&["SIZE"]), DataRequest::new(DatasetId::StockSwClassification,&["l1_code"])],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(&raw_id(i),if i<6 {9} else {0})],
            lookback: Lookback {trading_days: if i<6 {9} else {0}},
        }
    }
    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec(self.0)]
    }
    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_trade_disagreement_provider".into()
    }
    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        Ok(minute_compute_many(&[raw_id.to_string()], context, data)?
            .into_iter()
            .next())
    }
    fn minute_compute_many(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        minute_compute_many(ids, context, data)
    }
    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let id = raw_id(self.0);
        let panel = data.intraday_daily_raw_panel(&id)?;
        let raw = panel.column(&id)?;
        let raw = if self.0 < 6 {
            raw.ts(|v| ts_mean(v, 10, 1))?
        } else {
            raw
        };
        let result = neutralize_size_sector(&raw, &panel, data)?;
        Ok(result.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Default)]
struct Minute {
    open: Option<f64>,
    close: Option<f64>,
    vol: Option<f64>,
    amount: Option<f64>,
}
struct Stock {
    code: String,
    minute: Vec<Minute>,
    ret: Vec<Option<f64>>,
}
#[derive(Default)]
struct Daily {
    value: [Option<f64>; 10],
    counts: [[Counts; 2]; 2],
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|v| v.is_finite())
}
fn positive(v: Option<f64>) -> Option<f64> {
    finite(v).filter(|v| *v > 0.0)
}
fn ratio_return(close: Option<f64>, base: Option<f64>) -> Option<f64> {
    finite(Some(positive(close)? / positive(base)? - 1.0))
}

fn slot(time: &str) -> Option<usize> {
    let time = time.rsplit([' ', 'T']).next()?;
    let b = time.as_bytes();
    if b.len() < 8 || b[2] != b':' || b[5] != b':' || &b[6..8] != b"00" {
        return None;
    }
    let h = time.get(..2)?.parse::<usize>().ok()?;
    let m = time.get(3..5)?.parse::<usize>().ok()?;
    if h > 23 || m > 59 {
        return None;
    }
    let minute = h * 60 + m;
    match minute {
        571..=690 => Some(minute - 571),
        781..=900 => Some(minute - 781 + 120),
        _ => None,
    }
}

fn stock_returns(minute: &[Minute]) -> Vec<Option<f64>> {
    (0..240)
        .map(|t| {
            ratio_return(
                minute[t].close,
                if t == 0 {
                    minute[0].open
                } else {
                    minute[t - 1].close
                },
            )
        })
        .collect()
}

pub fn minute_compute_many(
    ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let mut requested = [false; 10];
    for i in 0..10 {
        requested[i] = ids.contains(&raw_id(i));
    }
    if !requested.iter().any(|v| *v) {
        return Ok(Vec::new());
    }
    let need_volume = requested
        .iter()
        .enumerate()
        .any(|(i, b)| *b && i != 2 && i != 3);
    let need_market = requested[2..].iter().any(|v| *v);
    let mut output: Vec<_> = (0..10)
        .filter(|&i| requested[i])
        .map(|i| {
            (
                i,
                IntradayDailyRawSeries {
                    spec: raw_spec(i),
                    values: Vec::new(),
                },
            )
        })
        .collect();
    for &date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, date) else {
            continue;
        };
        let code = table.required_utf8("ts_code")?;
        let time = table.required_utf8("trade_time")?;
        let open = table.required_f64_cast("open")?;
        let close = table.required_f64_cast("close")?;
        let vol = if need_volume {
            Some(table.required_f64_cast("vol")?)
        } else {
            None
        };
        let amount = if requested[4] {
            Some(table.required_f64_cast("amount")?)
        } else {
            None
        };
        let mut grouped = BTreeMap::<&str, Vec<Minute>>::new();
        for row in 0..table.len {
            let (Some(code), Some(time)) = (code[row].as_deref(), time[row].as_deref()) else {
                continue;
            };
            if is_bj_stock(code)
                || code.len() != 9
                || !code.as_bytes()[..6].iter().all(u8::is_ascii_digit)
                || !(code.ends_with(".SH") || code.ends_with(".SZ"))
            {
                continue;
            }
            let Some(t) = slot(time) else {
                continue;
            };
            grouped
                .entry(code)
                .or_insert_with(|| vec![Minute::default(); 240])[t] = Minute {
                open: positive(open[row]),
                close: positive(close[row]),
                vol: vol
                    .as_ref()
                    .and_then(|v| finite(v[row]))
                    .filter(|v| *v >= 0.0),
                amount: amount
                    .as_ref()
                    .and_then(|v| finite(v[row]))
                    .filter(|v| *v >= 0.0),
            };
        }
        let stocks: Vec<_> = grouped
            .into_iter()
            .map(|(code, minute)| Stock {
                code: code.to_string(),
                ret: stock_returns(&minute),
                minute,
            })
            .collect();
        let mut market = vec![None; 240];
        if need_market {
            for t in 0..240 {
                let mut sum = 0.0;
                let mut n = 0;
                for s in &stocks {
                    if let Some(r) = s.ret[t] {
                        sum += r;
                        n += 1;
                    }
                }
                if n > 0 {
                    market[t] = Some(sum / n as f64);
                }
            }
        }
        let mut daily: Vec<_> = stocks
            .par_iter()
            .map(|s| compute_stock(s, &market, &requested))
            .collect();
        for group in 0..2 {
            if !requested[8 + group] {
                continue;
            }
            let priors: Vec<_> = (0..2)
                .map(|g| prior(&daily.iter().map(|d| d.counts[group][g]).collect::<Vec<_>>()))
                .collect();
            if let (Some(a), Some(b)) = (priors[0], priors[1]) {
                for d in &mut daily {
                    d.value[8 + group] = bayes_onset(d.counts[group], [a, b]);
                }
            }
        }
        for (s, d) in stocks.iter().zip(daily) {
            for (i, series) in &mut output {
                series.values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: date,
                        ts_code: s.code.clone(),
                    },
                    value: finite(d.value[*i]),
                });
            }
        }
    }
    Ok(output.into_iter().map(|(_, s)| s).collect())
}

fn compute_stock(s: &Stock, market: &[Option<f64>], requested: &[bool; 10]) -> Daily {
    let mut out = Daily::default();
    let mut residual = vec![[None; 2]; 240];
    if requested[0] || requested[1] || requested[6..].iter().any(|v| *v) {
        let volumes: Vec<_> = s.minute.iter().filter_map(|p| p.vol).collect();
        if let Some(avg) = mean(&volumes).filter(|v| *v > 0.0) {
            let mut times = Vec::new();
            let mut x = Vec::new();
            let mut y = Vec::new();
            for t in 1..240 {
                let (Some(r), Some(v), Some(lag)) =
                    (s.ret[t], s.minute[t].vol, s.minute[t - 1].vol)
                else {
                    continue;
                };
                times.push(t);
                y.push(v / avg);
                x.push([
                    1.0,
                    r,
                    lag / avg,
                    if t < 30 { 1.0 } else { 0.0 },
                    if t >= 210 { 1.0 } else { 0.0 },
                ]);
            }
            for (j, q) in [0.2, 0.8].into_iter().enumerate() {
                if j == 0 && !requested[0] && !requested[6..].iter().any(|v| *v) {
                    continue;
                }
                if let Some(r) = quantile_residual(&x, &y, q) {
                    for (&t, v) in times.iter().zip(r) {
                        residual[t][j] = Some(v);
                    }
                }
            }
        }
        if requested[0] {
            out.value[0] = corr(
                &residual
                    .iter()
                    .filter_map(|r| Some((r[0]?, r[1]?)))
                    .collect::<Vec<_>>(),
            );
        }
        if requested[1] {
            out.value[1] = corr(
                &(0..240)
                    .filter_map(|t| Some((residual[t][1]?, s.ret[t]?)))
                    .collect::<Vec<_>>(),
            );
        }
    }
    if requested[2] || requested[3] {
        let morning: Vec<_> = (0..30)
            .filter_map(|t| Some((market[t]?, (s.ret[t]? - market[t]?).abs())))
            .collect();
        let mor = csad_fit(&morning);
        if requested[2] {
            out.value[2] = mor.map(|p| p.1);
        }
        if requested[3] {
            let after: Vec<_> = (30..240)
                .filter_map(|t| Some((market[t]?, (s.ret[t]? - market[t]?).abs())))
                .collect();
            out.value[3] = mor.zip(csad_fit(&after)).map(|(m, a)| a.0 - m.0);
        }
    }
    if requested[4..].iter().any(|v| *v) {
        let times: Vec<_> = (0..240)
            .filter(|&t| s.ret[t].is_some() && market[t].is_some())
            .collect();
        let pairs: Vec<_> = times
            .iter()
            .map(|&t| {
                (
                    market[t].unwrap(),
                    (s.ret[t].unwrap() - market[t].unwrap()).abs(),
                )
            })
            .collect();
        let Some(g) = sparse_gamma(&pairs) else {
            return out;
        };
        let mut gamma = vec![None; 240];
        for (&t, v) in times.iter().zip(g) {
            gamma[t] = Some(v);
        }
        let dispersion = |t: usize| gamma[t].is_some_and(|v| v > 0.0);
        if requested[4] {
            let vwap = |t: usize| finite(Some(s.minute[t].amount? / positive(s.minute[t].vol)?));
            let all: Vec<_> = (0..240).filter_map(vwap).collect();
            let subset: Vec<_> = (0..240)
                .filter(|&t| dispersion(t))
                .filter_map(vwap)
                .collect();
            out.value[4] = mean(&subset)
                .zip(mean(&all).filter(|v| *v > 0.0))
                .map(|(a, b)| a / b);
        }
        if requested[5] {
            out.value[5] = corr(
                &(0..240)
                    .filter(|&t| dispersion(t))
                    .filter_map(|t| Some((s.minute[t].vol?, s.ret[t]?.abs())))
                    .collect::<Vec<_>>(),
            );
        }
        for group in 0..2 {
            if requested[6 + group] || requested[8 + group] {
                out.counts[group] = onset(&gamma, &residual, group == 0);
                if requested[6 + group] {
                    out.value[6 + group] = basic_onset(out.counts[group]);
                }
            }
        }
    }
    out
}

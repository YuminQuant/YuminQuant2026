use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct PerformancePoint {
    pub factor_id: String,
    pub factor_date: i32,
    pub trade_date: Option<i32>,
    pub settle_date: Option<i32>,
    pub portfolio: String,
    pub return_value: Option<f64>,
    pub nav: Option<f64>,
    pub turnover: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FactorStatsDaily {
    pub factor_id: String,
    pub trade_date: i32,
    pub values: Vec<Option<f64>>,
    pub coverage: f64,
    pub inf_rate: f64,
}

#[derive(Clone, Debug)]
pub struct FactorStatsSummary {
    pub factor_id: String,
    pub scope: String,
    pub year: Option<i32>,
    pub observations: i64,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub p25: Option<f64>,
    pub median: Option<f64>,
    pub p75: Option<f64>,
    pub max: Option<f64>,
    pub coverage_mean: Option<f64>,
    pub inf_rate_mean: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct IcSummary {
    pub factor_id: String,
    pub scope: String,
    pub year: Option<i32>,
    pub horizon: Option<usize>,
    pub observations: i64,
    pub ic_mean: Option<f64>,
    pub ic_std: Option<f64>,
    pub icir: Option<f64>,
    pub ic_abs_mean: Option<f64>,
    pub ic_abs_std: Option<f64>,
    pub icir_abs: Option<f64>,
    pub rank_ic_mean: Option<f64>,
    pub rank_ic_std: Option<f64>,
    pub rank_icir: Option<f64>,
    pub rank_ic_abs_mean: Option<f64>,
    pub rank_ic_abs_std: Option<f64>,
    pub rank_icir_abs: Option<f64>,
    pub coverage_mean: Option<f64>,
    pub inf_rate_mean: Option<f64>,
}

pub fn summarize_factor_stats(rows: &[FactorStatsDaily]) -> Vec<FactorStatsSummary> {
    let mut grouped = BTreeMap::<(String, Option<i32>), Vec<&FactorStatsDaily>>::new();
    for row in rows {
        grouped
            .entry((row.factor_id.clone(), None))
            .or_default()
            .push(row);
        grouped
            .entry((row.factor_id.clone(), Some(row.trade_date / 10_000)))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(|((factor_id, year), rows)| {
            let values = rows
                .iter()
                .flat_map(|row| row.values.iter())
                .filter_map(|value| value.filter(|value| value.is_finite()))
                .collect::<Vec<_>>();
            let coverage = rows.iter().map(|row| row.coverage).collect::<Vec<_>>();
            let inf_rate = rows.iter().map(|row| row.inf_rate).collect::<Vec<_>>();
            FactorStatsSummary {
                factor_id,
                scope: year.map_or_else(|| "full".to_string(), |_| "year".to_string()),
                year,
                observations: values.len() as i64,
                mean: mean(&values),
                std: std_dev(&values),
                min: values.iter().copied().reduce(f64::min),
                p25: quantile(values.clone(), 0.25),
                median: quantile(values.clone(), 0.5),
                p75: quantile(values.clone(), 0.75),
                max: values.iter().copied().reduce(f64::max),
                coverage_mean: mean(&coverage),
                inf_rate_mean: mean(&inf_rate),
            }
        })
        .collect()
}

pub fn summarize_ic(
    factor_id: &str,
    year: Option<i32>,
    horizon: Option<usize>,
    ic: &[Option<f64>],
    rank_ic: &[Option<f64>],
    coverage: &[f64],
    inf_rate: &[f64],
) -> IcSummary {
    let ic_values = finite_values(ic);
    let rank_values = finite_values(rank_ic);
    let ic_abs = ic_values
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let rank_abs = rank_values
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    IcSummary {
        factor_id: factor_id.to_string(),
        scope: year.map_or_else(|| "full".to_string(), |_| "year".to_string()),
        year,
        horizon,
        observations: ic_values.len() as i64,
        ic_mean: mean(&ic_values),
        ic_std: std_dev(&ic_values),
        icir: ratio(mean(&ic_values), std_dev(&ic_values)),
        ic_abs_mean: mean(&ic_abs),
        ic_abs_std: std_dev(&ic_abs),
        icir_abs: ratio(mean(&ic_abs), std_dev(&ic_abs)),
        rank_ic_mean: mean(&rank_values),
        rank_ic_std: std_dev(&rank_values),
        rank_icir: ratio(mean(&rank_values), std_dev(&rank_values)),
        rank_ic_abs_mean: mean(&rank_abs),
        rank_ic_abs_std: std_dev(&rank_abs),
        rank_icir_abs: ratio(mean(&rank_abs), std_dev(&rank_abs)),
        coverage_mean: mean(coverage),
        inf_rate_mean: mean(inf_rate),
    }
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

fn finite_values(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .collect()
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn std_dev(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt())
}

fn quantile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    if values.len() == 1 {
        return values.first().copied();
    }
    let pos = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        values.get(lower).copied()
    } else {
        let weight = pos - lower as f64;
        Some(values[lower] * (1.0 - weight) + values[upper] * weight)
    }
}

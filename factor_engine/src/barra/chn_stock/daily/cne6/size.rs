use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::operators::{cs_neutralize_regression, cs_winsorize_quantile};

pub struct StockDailyBarraCne6Size;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.2.0";

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Size)
}

impl BarraExposure for StockDailyBarraCne6Size {
    fn family_id(&self) -> &'static str {
        "SIZE"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            size_spec(
                "Size",
                "CNE6 Size secondary exposure",
                "log(circ_mv), winsorized at 1%/99% and z-scored with sqrt(circ_mv) weights.",
            ),
            size_spec(
                "Mid_Cap",
                "CNE6 Mid_Cap secondary exposure",
                "Size cubed, WLS residualized against Size with sqrt(circ_mv) weights, then winsorized and weighted z-scored.",
            ),
            size_spec(
                "SIZE",
                "CNE6 SIZE style exposure",
                "Equal-weight composite of Size and Mid_Cap, then z-scored with sqrt(circ_mv) weights.",
            ),
        ]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let circ_mv = panel.column("circ_mv")?;
        let sqrt_mv = circ_mv.map_values(sqrt_market_value);
        let raw_size = circ_mv
            .map_values(|value| clean(value).and_then(|value| (value > 0.0).then_some(value.ln())));
        let size = raw_size.cs_binary(&sqrt_mv, standardize_cross_section_weighted)?;

        let mid_raw = size.map_values(|value| clean(value).map(|value| value.powi(3)));
        let mid_residual = mid_raw.cs_ternary(&size, &sqrt_mv, |mid, size, weight| {
            cs_neutralize_regression(mid, &[size], None, Some(weight))
        })?;
        let mid_cap = mid_residual.cs_binary(&sqrt_mv, standardize_cross_section_weighted)?;

        let composite_raw = size.zip_binary(&mid_cap, |size, mid_cap| {
            match (clean(size), clean(mid_cap)) {
                (Some(size), Some(mid_cap)) => Some((size + mid_cap) / 2.0),
                _ => None,
            }
        })?;
        let composite_size =
            composite_raw.cs_binary(&sqrt_mv, standardize_cross_section_weighted)?;

        let specs = self.specs();
        Ok(vec![
            size.to_barra_series(specs[0].clone()),
            mid_cap.to_barra_series(specs[1].clone()),
            composite_size.to_barra_series(specs[2].clone()),
        ])
    }
}

fn size_spec(id: &str, name: &str, description: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "size", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: description.to_string(),
        dependencies: vec![DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"])],
        lookback: Lookback { trading_days: 0 },
    }
}

fn standardize_cross_section_weighted(
    values: &[Option<f64>],
    weights: &[Option<f64>],
) -> Vec<Option<f64>> {
    let winsorized = cs_winsorize_quantile(values, 0.01, 0.99);
    weighted_zscore(&winsorized, weights)
}

fn weighted_zscore(values: &[Option<f64>], weights: &[Option<f64>]) -> Vec<Option<f64>> {
    if values.len() != weights.len() {
        return vec![None; values.len()];
    }
    let mut rows = Vec::new();
    let mut weight_sum = 0.0;
    let mut weighted_sum = 0.0;
    for idx in 0..values.len() {
        let (Some(value), Some(weight)) = (clean(values[idx]), clean(weights[idx])) else {
            continue;
        };
        if !value.is_finite() || !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        rows.push((idx, value, weight));
        weight_sum += weight;
        weighted_sum += weight * value;
    }
    if rows.is_empty() || weight_sum <= f64::EPSILON {
        return vec![None; values.len()];
    }
    let mean = weighted_sum / weight_sum;
    let variance = rows
        .iter()
        .map(|(_, value, weight)| weight * (value - mean).powi(2))
        .sum::<f64>()
        / weight_sum;
    if variance <= f64::EPSILON {
        return vec![None; values.len()];
    }
    let std = variance.sqrt();
    let mut output = vec![None; values.len()];
    for (idx, value, _) in rows {
        output[idx] = Some((value - mean) / std);
    }
    output
}

fn sqrt_market_value(value: Option<f64>) -> Option<f64> {
    clean(value).and_then(|value| (value > 0.0).then_some(value.sqrt()))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::operators::cs_winsorize_quantile;

    use super::{sqrt_market_value, standardize_cross_section_weighted, StockDailyBarraCne6Size};
    use crate::barra::BarraExposure;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn cne6_size_family_registers_three_exposures() {
        let exposure = StockDailyBarraCne6Size;
        let specs = exposure.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["Size", "Mid_Cap", "SIZE"]);
        assert!(specs.iter().all(|spec| spec.model == "CNE6"));
    }

    #[test]
    fn two_sided_winsorize_clamps_quantile_tails() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0), Some(100.0)];
        let output = cs_winsorize_quantile(&values, 0.25, 0.75);

        assert_close(output[0].unwrap(), 1.75);
        assert_close(output[3].unwrap(), 27.25);
    }

    #[test]
    fn standardization_keeps_weighted_zero_mean_after_winsorize() {
        let weights = [Some(1.0), Some(3.0), Some(6.0), Some(10.0)];
        let output = standardize_cross_section_weighted(
            &[Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
            &weights,
        );
        let weighted_mean = output
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| value.unwrap() * weight.unwrap())
            .sum::<f64>()
            / weights.iter().map(|value| value.unwrap()).sum::<f64>();

        assert!(weighted_mean.abs() < 1e-12);
    }

    #[test]
    fn sqrt_market_value_rejects_non_positive_values() {
        assert_close(sqrt_market_value(Some(16.0)).unwrap(), 4.0);
        assert_eq!(sqrt_market_value(Some(0.0)), None);
        assert_eq!(sqrt_market_value(Some(-1.0)), None);
    }
}

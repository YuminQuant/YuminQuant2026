use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::operators::{cs_neutralize_regression, cs_winsorize_quantile, cs_zscore};

pub struct StockDailyBarraCne6Size;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.1.0";

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Size)
}

impl BarraExposure for StockDailyBarraCne6Size {
    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            size_spec(
                "Size",
                "CNE6 Size secondary exposure",
                "log(total_mv), winsorized at 1%/99% and equal-weight z-scored.",
            ),
            size_spec(
                "Mid_Cap",
                "CNE6 Mid_Cap secondary exposure",
                "Size cubed, WLS residualized against Size with total_mv weights, then winsorized and z-scored.",
            ),
            size_spec(
                "SIZE",
                "CNE6 SIZE style exposure",
                "Equal-weight composite of Size and Mid_Cap, then equal-weight z-scored.",
            ),
        ]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let total_mv = panel.column("total_mv")?;
        let raw_size = total_mv
            .map_values(|value| clean(value).and_then(|value| (value > 0.0).then_some(value.ln())));
        let size = raw_size.cs(standardize_cross_section)?;

        let mid_raw = size.map_values(|value| clean(value).map(|value| value.powi(3)));
        let mid_residual = mid_raw.cs_ternary(&size, &total_mv, |mid, size, weight| {
            cs_neutralize_regression(mid, &[size], None, Some(weight))
        })?;
        let mid_cap = mid_residual.cs(standardize_cross_section)?;

        let composite_raw = size.zip_binary(&mid_cap, |size, mid_cap| {
            match (clean(size), clean(mid_cap)) {
                (Some(size), Some(mid_cap)) => Some((size + mid_cap) / 2.0),
                _ => None,
            }
        })?;
        let composite_size = composite_raw.cs(cs_zscore)?;

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
        dependencies: vec![DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"])],
        lookback: Lookback { trading_days: 0 },
    }
}

fn standardize_cross_section(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(&cs_winsorize_quantile(values, 0.01, 0.99))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::operators::cs_winsorize_quantile;

    use super::{standardize_cross_section, StockDailyBarraCne6Size};
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
    fn standardization_keeps_zero_mean_after_winsorize() {
        let output = standardize_cross_section(&[Some(10.0), Some(20.0), Some(30.0), Some(40.0)]);
        let mean = output.iter().map(|value| value.unwrap()).sum::<f64>() / 4.0;

        assert!(mean.abs() < 1e-12);
    }
}

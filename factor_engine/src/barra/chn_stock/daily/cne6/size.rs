use crate::barra::common::standardize_panel_industry_filled_weighted;
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::operators::cs_neutralize_regression;

pub struct StockDailyBarraCne6Size;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.3.0";

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
        let size = standardize_panel_industry_filled_weighted(&raw_size, &sqrt_mv, data)?;

        let mid_raw = size.map_values(|value| clean(value).map(|value| value.powi(3)));
        let mid_residual = mid_raw.cs_ternary(&size, &sqrt_mv, |mid, size, weight| {
            cs_neutralize_regression(mid, &[size], None, Some(weight))
        })?;
        let mid_cap = standardize_panel_industry_filled_weighted(&mid_residual, &sqrt_mv, data)?;

        let composite_raw = size.zip_binary(&mid_cap, |size, mid_cap| {
            match (clean(size), clean(mid_cap)) {
                (Some(size), Some(mid_cap)) => Some((size + mid_cap) / 2.0),
                _ => None,
            }
        })?;
        let composite_size =
            standardize_panel_industry_filled_weighted(&composite_raw, &sqrt_mv, data)?;

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
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        lookback: Lookback { trading_days: 0 },
    }
}

fn sqrt_market_value(value: Option<f64>) -> Option<f64> {
    clean(value).and_then(|value| (value > 0.0).then_some(value.sqrt()))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

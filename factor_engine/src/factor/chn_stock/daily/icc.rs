use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::factor::Factor;

pub struct StockDailyIcc;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIcc)
}

impl Factor for StockDailyIcc {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "icc".into(),
            aliases: vec!["ICC".into(), "Implied Cost of Capital".into()],
            name: "Implied Cost of Capital".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".into(),
            tags: ["DFZQ", "analyst", "consensus", "valuation", "daily", "neutralize", "barra", "size", "sector"]
                .into_iter().map(str::to_string).collect(),
            description: "Zero-dividend Easton raw ICC: sqrt((con_eps_fy2-con_eps_fy1)/close). EPS and unadjusted close are yuan/share; raw is a decimal rate. Uses the consensus dataset's PIT fiscal-year anchor. Requires finite EPS, strictly positive EPS increase and close; negative EPS levels are allowed. Stock universe panel, excludes BJ. Final output is the daily SW L1 industry and Barra SIZE regression residual, which may be negative; Missing raw gets zero residual only with valid SIZE and a fitted industry; missing SIZE/industry stays null. No rank transform, winsorization or final zscore.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockConsensus, &["con_eps_fy1", "con_eps_fy2"]).with_panel_only(),
                DataRequest::new(DatasetId::StockDailyPv, &["close"]).with_panel_only(),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]).with_panel_only(),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let raw = raw_icc(data)?;
        Ok(
            finish_analyst(&raw, data.stock_universe_panel()?, data)?
                .to_factor_series(self.spec()),
        )
    }
}

fn raw_icc(data: &DataPool) -> Result<PanelColumn> {
    let panel = data.stock_universe_panel()?;
    let consensus = data.daily_panel(DatasetId::StockConsensus)?;
    let eps1 = panel.column_from_panel(consensus, "con_eps_fy1")?;
    let eps2 = panel.column_from_panel(consensus, "con_eps_fy2")?;
    let close = panel.column_from_panel(data.daily_panel(DatasetId::StockDailyPv)?, "close")?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            let offset = day * n + i;
            if panel.is_present_offset(offset) && !is_bj_stock(code) {
                values[offset] = icc(
                    eps1.values()[offset],
                    eps2.values()[offset],
                    close.values()[offset],
                );
            }
        }
    }
    panel.column_from_values(values)
}

fn icc(eps1: Option<f64>, eps2: Option<f64>, close: Option<f64>) -> Option<f64> {
    let eps1 = eps1.filter(|v| v.is_finite())?;
    let eps2 = eps2.filter(|v| v.is_finite())?;
    let close = close.filter(|v| v.is_finite() && *v > 0.0)?;
    let increase = eps2 - eps1;
    if !increase.is_finite() || increase <= 0.0 {
        return None;
    }
    let result = (increase / close).sqrt();
    (result.is_finite() && result > 0.0).then_some(result)
}

// Fill only the final signal; financial and time-series inputs retain their missing values.
fn finish_analyst(raw: &PanelColumn, panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let size = panel.column_from_panel(data.daily_panel(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?, ClassificationLevel::Sector,
    )?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let start = day * n;
        let mut groups = sector.groups_for(date, panel.instruments());
        for (i, code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(code) || !panel.is_present_offset(start + i) {
                groups[i] = None;
            }
        }
        let residual = neutralize_and_fill_missing(
            &raw.values()[start..start + n],
            &size.values()[start..start + n],
            &groups,
        );
        values[start..start + n].copy_from_slice(&residual);
    }
    panel.column_from_values(values)
}

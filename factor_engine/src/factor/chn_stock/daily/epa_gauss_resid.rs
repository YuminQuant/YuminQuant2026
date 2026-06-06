use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gaussian_financial::gaussian_residual;
use crate::factor::common::stock_daily_ops::mask_bj;
use crate::factor::common::vector::clean;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::{ts_ir, ts_zscore};

const VERSION: &str = "0.1.0";
const EPA_Z_WINDOW: usize = 240;
const EPA_Z_MIN_PERIODS: usize = 2;
const RET_IR_WINDOW: usize = 126;
const RET_IR_MIN_PERIODS: usize = 2;
const LOOKBACK: usize = 252;

pub struct StockDailyEpaGaussResid;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyEpaGaussResid)
}

impl Factor for StockDailyEpaGaussResid {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "epa_gauss_resid".to_string(),
            aliases: vec!["EPA Gaussian Residual".to_string()],
            name: "epa_gauss_resid".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DFZQ/DBZQ standalone EPA Gaussian-rank reconstruction factor. It reconstructs daily netprofit_ttm from daily pe_ttm and total_mv, Gaussian-rank residualizes it on daily total_mv, applies 240-day zscore, then neutralizes VOLATILITY, VALUE, GROWTH, 126-day return IR and sector while excluding BJ stocks.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["pe_ttm", "total_mv"]),
                DataRequest::new(
                    DatasetId::StockBarraDaily,
                    &["VOLATILITY", "VALUE", "GROWTH"],
                ),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: LOOKBACK,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic = data.daily(DatasetId::StockDailyBasic)?;
        let pe_ttm = panel.column_from_table(basic, "pe_ttm")?;
        let total_mv = panel.column_from_table(basic, "total_mv")?;
        let total_mv = mask_bj(&total_mv, &panel)?;
        let netprofit_ttm = reconstructed_netprofit_ttm(&pe_ttm, &total_mv)?;
        let netprofit_ttm = mask_bj(&netprofit_ttm, &panel)?;
        let ep_raw = gaussian_residual(&netprofit_ttm, &[&total_mv])?;
        let ep_z240 = ep_raw.ts(|values| ts_zscore(values, EPA_Z_WINDOW, EPA_Z_MIN_PERIODS))?;
        let factor = neutralize_epa(&ep_z240, &panel, data)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "DFZQ",
        "DBZQ",
        "DWZQ",
        "financial",
        "fundamental",
        "valuation",
        "epa",
        "gaussian_rank",
        "residual",
        "neutralize",
        "barra",
        "style",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn reconstructed_netprofit_ttm(
    pe_ttm: &PanelColumn,
    total_mv: &PanelColumn,
) -> Result<PanelColumn> {
    pe_ttm.zip_binary(total_mv, netprofit_from_pe_and_mv)
}

fn netprofit_from_pe_and_mv(pe_ttm: Option<f64>, total_mv: Option<f64>) -> Option<f64> {
    match (clean(pe_ttm), clean(total_mv).filter(|value| *value > 0.0)) {
        (Some(pe_ttm), Some(total_mv)) if pe_ttm.abs() > f64::EPSILON => {
            let value = total_mv / pe_ttm;
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn neutralize_epa(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let barra = data.daily(DatasetId::StockBarraDaily)?;
    let volatility = panel.column_from_table(barra, "VOLATILITY")?;
    let value = panel.column_from_table(barra, "VALUE")?;
    let growth = panel.column_from_table(barra, "GROWTH")?;
    let ret_ir = adjusted_return_ir_126(panel, data)?;
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let masked = mask_bj(values, panel)?;
    masked.cs_neutralize_regression_by_group(
        &[&volatility, &value, &growth, &ret_ir],
        None,
        |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
    )
}

fn adjusted_return_ir_126(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, |close, adj_factor| {
        match (clean(close), clean(adj_factor)) {
            (Some(close), Some(adj_factor)) => Some(close * adj_factor),
            _ => None,
        }
    })?;
    let returns = adj_close.ts(adjacent_returns)?;
    returns.ts(|values| ts_ir(values, RET_IR_WINDOW, RET_IR_MIN_PERIODS))
}

fn adjacent_returns(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        let (Some(current), Some(previous)) = (clean(values[idx]), clean(values[idx - 1])) else {
            continue;
        };
        if previous.abs() <= f64::EPSILON {
            continue;
        }
        let value = current / previous - 1.0;
        if value.is_finite() {
            output[idx] = Some(value);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epa_reconstructs_netprofit_ttm_from_pe_and_market_value() {
        assert_eq!(
            netprofit_from_pe_and_mv(Some(10.0), Some(1000.0)),
            Some(100.0)
        );
        assert_eq!(netprofit_from_pe_and_mv(Some(0.0), Some(1000.0)), None);
        assert_eq!(netprofit_from_pe_and_mv(Some(10.0), Some(0.0)), None);
    }

    #[test]
    fn epa_spec_uses_daily_valuation_not_financial_statements() {
        let spec = StockDailyEpaGaussResid.spec();
        assert_eq!(spec.id, "epa_gauss_resid");
        assert!(spec.tags.contains(&"DFZQ".to_string()));
        assert!(spec.tags.contains(&"DBZQ".to_string()));
        assert!(spec.dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockDailyBasic
                && request.columns.iter().any(|column| column == "pe_ttm")
                && request.columns.iter().any(|column| column == "total_mv")
        }));
        assert!(!spec.dependencies.iter().any(|request| {
            matches!(
                request.dataset,
                DatasetId::StockIncome
                    | DatasetId::StockBalanceSheet
                    | DatasetId::StockCashFlow
                    | DatasetId::StockDividend
            )
        }));
    }
}

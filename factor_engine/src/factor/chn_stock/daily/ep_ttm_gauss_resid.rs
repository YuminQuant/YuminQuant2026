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

const VERSION: &str = "0.1.0";

pub struct StockDailyEpTtmGaussResid;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyEpTtmGaussResid)
}

impl Factor for StockDailyEpTtmGaussResid {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ep_ttm_gauss_resid".to_string(),
            aliases: vec!["EPTTM Gaussian Residual".to_string()],
            name: "ep_ttm_gauss_resid".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DFZQ/DBZQ standalone EPTTM Gaussian-rank reconstruction factor. It reconstructs daily netprofit_ttm from daily pe_ttm and total_mv, Gaussian-rank residualizes it on daily total_mv, applies sector neutralization, and excludes BJ stocks.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["pe_ttm", "total_mv"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
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
        let raw = gaussian_residual(&netprofit_ttm, &[&total_mv])?;
        let factor = neutralize_sector_only(&raw, &panel, data)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "DFZQ",
        "DBZQ",
        "financial",
        "fundamental",
        "valuation",
        "ep_ttm",
        "gaussian_rank",
        "residual",
        "neutralize",
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

fn neutralize_sector_only(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let masked = mask_bj(values, panel)?;
    masked.cs_neutralize_regression_by_group(&[], None, |trade_date, ts_codes| {
        sector_map.groups_for(trade_date, ts_codes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_ttm_reconstructs_netprofit_ttm_from_pe_and_market_value() {
        assert_eq!(
            netprofit_from_pe_and_mv(Some(10.0), Some(1000.0)),
            Some(100.0)
        );
        assert_eq!(netprofit_from_pe_and_mv(Some(0.0), Some(1000.0)), None);
        assert_eq!(netprofit_from_pe_and_mv(Some(10.0), Some(0.0)), None);
    }

    #[test]
    fn ep_ttm_spec_uses_daily_valuation_not_financial_statements() {
        let spec = StockDailyEpTtmGaussResid.spec();
        assert_eq!(spec.id, "ep_ttm_gauss_resid");
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

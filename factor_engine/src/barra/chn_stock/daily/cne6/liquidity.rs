use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::operators::{cs_winsorize_quantile, cs_zscore};

pub struct StockDailyBarraCne6Liquidity;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.1.0";
const TRADING_DAYS_PER_MONTH: usize = 21;
const TRADING_DAYS_PER_YEAR: usize = 252;
const ATVR_HALF_LIFE: f64 = 63.0;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Liquidity)
}

impl BarraExposure for StockDailyBarraCne6Liquidity {
    fn family_id(&self) -> &'static str {
        "LIQUIDITY"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            liquidity_spec(
                "Monthly_Share_Turnover",
                &["STOM"],
                "CNE6 monthly share turnover",
                "Log of the most recent 21 trading days' free-float share turnover sum.",
                20,
            ),
            liquidity_spec(
                "Quarterly_Share_Turnover",
                &["STOQ"],
                "CNE6 quarterly share turnover",
                "Log of the average of the most recent three 21-day monthly turnover sums.",
                62,
            ),
            liquidity_spec(
                "Annual_Share_Turnover",
                &["STOA"],
                "CNE6 annual share turnover",
                "Log of the average of the most recent twelve 21-day monthly turnover sums.",
                251,
            ),
            liquidity_spec(
                "Annualized_Traded_Value_Ratio",
                &["ATVR"],
                "CNE6 annualized traded value ratio",
                "Annualized exponentially weighted daily turnover over 252 trading days with half-life 63.",
                251,
            ),
            liquidity_spec(
                "LIQUIDITY",
                &[],
                "CNE6 LIQUIDITY style exposure",
                "Equal-weight composite of the four standardized liquidity sub-exposures, then z-scored.",
                251,
            ),
        ]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let turnover = panel.column("turnover_rate_f")?.map_values(|value| {
            clean(value).and_then(|value| (value >= 0.0).then_some(value / 100.0))
        });

        let monthly_raw = turnover.ts(stom_raw)?;
        let monthly = monthly_raw.cs(standardize_cross_section)?;
        let quarterly = monthly_raw
            .ts(|values| stoq_or_stoa_from_stom(values, 3))?
            .cs(standardize_cross_section)?;
        let annual = monthly_raw
            .ts(|values| stoq_or_stoa_from_stom(values, 12))?
            .cs(standardize_cross_section)?;
        let atvr = turnover
            .ts(annualized_traded_value_ratio)?
            .cs(standardize_cross_section)?;

        let composite_raw = monthly.zip_quaternary(
            &quarterly,
            &annual,
            &atvr,
            |monthly, quarterly, annual, atvr| match (
                clean(monthly),
                clean(quarterly),
                clean(annual),
                clean(atvr),
            ) {
                (Some(monthly), Some(quarterly), Some(annual), Some(atvr)) => {
                    Some((monthly + quarterly + annual + atvr) / 4.0)
                }
                _ => None,
            },
        )?;
        let liquidity = composite_raw.cs(cs_zscore)?;

        let specs = self.specs();
        Ok(vec![
            monthly.to_barra_series(specs[0].clone()),
            quarterly.to_barra_series(specs[1].clone()),
            annual.to_barra_series(specs[2].clone()),
            atvr.to_barra_series(specs[3].clone()),
            liquidity.to_barra_series(specs[4].clone()),
        ])
    }
}

fn liquidity_spec(
    id: &str,
    aliases: &[&str],
    name: &str,
    description: &str,
    lookback: usize,
) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: aliases.iter().map(|value| value.to_string()).collect(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: [
            "barra",
            "cne6",
            "style",
            "liquidity",
            "turnover",
            "daily",
            "stock",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect(),
        description: description.to_string(),
        dependencies: vec![DataRequest::new(
            DatasetId::StockDailyBasic,
            &["turnover_rate_f"],
        )],
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

fn stom_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let Some(start) = idx.checked_sub(TRADING_DAYS_PER_MONTH - 1) else {
            continue;
        };
        let mut sum = 0.0;
        let mut count = 0;
        for value in &values[start..=idx] {
            let Some(value) = clean(*value) else {
                continue;
            };
            sum += value;
            count += 1;
        }
        if count == TRADING_DAYS_PER_MONTH && sum > 0.0 {
            output[idx] = Some(sum.ln());
        }
    }
    output
}

fn stoq_or_stoa_from_stom(stom: &[Option<f64>], months: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; stom.len()];
    if months == 0 {
        return output;
    }
    for idx in 0..stom.len() {
        let mut sum_month_turnover = 0.0;
        let mut count = 0;
        let mut valid = true;
        for month_idx in 0..months {
            let Some(stom_idx) = idx.checked_sub(month_idx * TRADING_DAYS_PER_MONTH) else {
                valid = false;
                break;
            };
            let Some(value) = clean(stom[stom_idx]) else {
                valid = false;
                break;
            };
            sum_month_turnover += value.exp();
            count += 1;
        }
        if valid && count == months && sum_month_turnover > 0.0 {
            output[idx] = Some((sum_month_turnover / months as f64).ln());
        }
    }
    output
}

fn annualized_traded_value_ratio(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        if idx + 1 < TRADING_DAYS_PER_YEAR {
            continue;
        }
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut count = 0;
        for lag in 0..TRADING_DAYS_PER_YEAR {
            let Some(value) = clean(values[idx - lag]) else {
                continue;
            };
            let weight = 0.5_f64.powf(lag as f64 / ATVR_HALF_LIFE);
            weighted_sum += weight * value;
            weight_sum += weight;
            count += 1;
        }
        if count == TRADING_DAYS_PER_YEAR && weight_sum > 0.0 {
            output[idx] = Some(weighted_sum / weight_sum * TRADING_DAYS_PER_YEAR as f64);
        }
    }
    output
}

fn standardize_cross_section(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(&cs_winsorize_quantile(values, 0.01, 0.99))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::{
        annualized_traded_value_ratio, stom_raw, stoq_or_stoa_from_stom,
        StockDailyBarraCne6Liquidity, TRADING_DAYS_PER_YEAR,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn cne6_liquidity_family_registers_sub_exposures_and_composite() {
        let exposure = StockDailyBarraCne6Liquidity;
        let specs = exposure.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "Monthly_Share_Turnover",
                "Quarterly_Share_Turnover",
                "Annual_Share_Turnover",
                "Annualized_Traded_Value_Ratio",
                "LIQUIDITY"
            ]
        );
        assert!(specs.iter().all(|spec| spec.model == "CNE6"));
    }

    #[test]
    fn monthly_turnover_uses_complete_21_day_blocks() {
        let values = vec![Some(1.0); 42];
        let stom = stom_raw(&values);
        let output = stoq_or_stoa_from_stom(&stom, 2);

        assert_eq!(output[40], None);
        assert_close(output[41].unwrap(), 21.0_f64.ln());
    }

    #[test]
    fn atvr_requires_full_year_and_annualizes_constant_turnover() {
        let values = vec![Some(2.0); TRADING_DAYS_PER_YEAR];
        let output = annualized_traded_value_ratio(&values);

        assert_eq!(output[TRADING_DAYS_PER_YEAR - 2], None);
        assert_close(output[TRADING_DAYS_PER_YEAR - 1].unwrap(), 2.0 * 252.0);
    }
}

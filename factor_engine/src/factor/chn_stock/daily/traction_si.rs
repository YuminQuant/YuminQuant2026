use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{
    adjusted_20d_return, is_bj_stock, neutralize_ret20_size_sector,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MIN_PAIR_DAYS: u8 = 10;

pub struct StockDailyTractionSi;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTractionSi)
}

impl Factor for StockDailyTractionSi {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "traction_si".to_string(),
            aliases: vec!["Traction-SI".to_string(), "ExpAve".to_string()],
            name: "traction_si".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ small-order moneyflow cross-sectional network traction factor. It builds a 20-day same-direction small-order net-flow network, computes association-weighted peer Ret20 ExpAve, and neutralizes by Ret20, Barra SIZE, and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_sm_amount", "sell_sm_amount"],
                ),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let buy = panel.column_from_table(moneyflow, "buy_sm_amount")?;
        let sell = panel.column_from_table(moneyflow, "sell_sm_amount")?;
        let small_direction = buy.zip_binary(&sell, small_flow_direction)?;
        let ret20 = adjusted_20d_return(data, &panel)?;
        let expave = traction_expave(&small_direction, &ret20, &panel)?;
        let factor = neutralize_ret20_size_sector(&expave, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "KYZQ",
        "cs_network",
        "moneyflow",
        "small_order",
        "network",
        "ret20",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn small_flow_direction(buy: Option<f64>, sell: Option<f64>) -> Option<f64> {
    let (Some(buy), Some(sell)) = (clean(buy), clean(sell)) else {
        return None;
    };
    let net = buy - sell;
    if net > 0.0 {
        Some(1.0)
    } else if net < 0.0 {
        Some(-1.0)
    } else {
        None
    }
}

fn traction_expave(
    directions: &PanelColumn,
    associated_returns: &PanelColumn,
    panel: &DailyPanel,
) -> Result<PanelColumn> {
    let code_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let eligible = eligible_instruments(panel);
    let mut output = vec![None; panel.shape_len()];
    let mut state = PairCountState::new(code_count);

    for date_idx in 0..date_count {
        if date_idx >= WINDOW {
            let remove_offset = (date_idx - WINDOW) * code_count;
            state.remove_day(
                &day_directions(directions.values(), remove_offset, code_count),
                &eligible,
            );
        }

        let add_offset = date_idx * code_count;
        state.add_day(
            &day_directions(directions.values(), add_offset, code_count),
            &eligible,
        );

        if date_idx + 1 < WINDOW {
            continue;
        }

        let returns = &associated_returns.values()[add_offset..add_offset + code_count];
        let expave = state.expave(returns, &eligible);
        for code_idx in 0..code_count {
            output[add_offset + code_idx] = expave[code_idx];
        }
    }

    panel.column_from_values(output)
}

fn eligible_instruments(panel: &DailyPanel) -> Vec<bool> {
    panel
        .instruments()
        .iter()
        .map(|ts_code| !is_bj_stock(ts_code))
        .collect()
}

fn day_directions(values: &[Option<f64>], offset: usize, len: usize) -> Vec<i8> {
    values[offset..offset + len]
        .iter()
        .map(|value| match clean(*value) {
            Some(value) if value > 0.0 => 1,
            Some(value) if value < 0.0 => -1,
            _ => 0,
        })
        .collect()
}

struct PairCountState {
    code_count: usize,
    same_count: Vec<u8>,
    valid_count: Vec<u8>,
}

impl PairCountState {
    fn new(code_count: usize) -> Self {
        let pair_count = code_count.saturating_mul(code_count.saturating_sub(1)) / 2;
        Self {
            code_count,
            same_count: vec![0; pair_count],
            valid_count: vec![0; pair_count],
        }
    }

    fn add_day(&mut self, directions: &[i8], eligible: &[bool]) {
        self.update_day(directions, eligible, 1);
    }

    fn remove_day(&mut self, directions: &[i8], eligible: &[bool]) {
        self.update_day(directions, eligible, -1);
    }

    fn update_day(&mut self, directions: &[i8], eligible: &[bool], delta: i8) {
        if self.code_count < 2 {
            return;
        }
        for left in 0..self.code_count - 1 {
            if !eligible[left] || directions[left] == 0 {
                continue;
            }
            let mut pair_idx = pair_index(left, left + 1, self.code_count);
            for right in left + 1..self.code_count {
                if eligible[right] && directions[right] != 0 {
                    update_counter(&mut self.valid_count[pair_idx], delta);
                    if directions[left] == directions[right] {
                        update_counter(&mut self.same_count[pair_idx], delta);
                    }
                }
                pair_idx += 1;
            }
        }
    }

    fn expave(&self, associated_returns: &[Option<f64>], eligible: &[bool]) -> Vec<Option<f64>> {
        let mut numerator = vec![0.0; self.code_count];
        let mut denominator = vec![0.0; self.code_count];

        if self.code_count >= 2 {
            for left in 0..self.code_count - 1 {
                if !eligible[left] {
                    continue;
                }
                let left_return = clean(associated_returns[left]);
                let mut pair_idx = pair_index(left, left + 1, self.code_count);
                for right in left + 1..self.code_count {
                    if eligible[right] {
                        let valid = self.valid_count[pair_idx];
                        let same = self.same_count[pair_idx];
                        if valid >= MIN_PAIR_DAYS && same > 0 {
                            let weight = same as f64 / valid as f64;
                            if let Some(right_return) = clean(associated_returns[right]) {
                                numerator[left] += weight * right_return;
                                denominator[left] += weight;
                            }
                            if let Some(left_return) = left_return {
                                numerator[right] += weight * left_return;
                                denominator[right] += weight;
                            }
                        }
                    }
                    pair_idx += 1;
                }
            }
        }

        numerator
            .into_iter()
            .zip(denominator)
            .map(|(num, den)| {
                if den > f64::EPSILON {
                    let value = num / den;
                    value.is_finite().then_some(value)
                } else {
                    None
                }
            })
            .collect()
    }
}

fn update_counter(value: &mut u8, delta: i8) {
    if delta > 0 {
        *value = value.saturating_add(delta as u8);
    } else {
        *value = value.saturating_sub((-delta) as u8);
    }
}

fn pair_index(left: usize, right: usize, code_count: usize) -> usize {
    debug_assert!(left < right);
    debug_assert!(right < code_count);
    left * (2 * code_count - left - 1) / 2 + (right - left - 1)
}

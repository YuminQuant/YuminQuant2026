use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::factor::Factor;
use std::collections::VecDeque;

const WINDOW: usize = 120;
const TOP: usize = 80;
const HALF_LIFE: f64 = 10.0;

#[derive(Clone, Copy)]
pub enum Side {
    Winner,
    Loser,
}
impl Side {
    fn id(self) -> &'static str {
        match self {
            Self::Winner => "dwf",
            Self::Loser => "dlf",
        }
    }
}
pub struct DailyLeaderboard(pub Side);
impl Factor for DailyLeaderboard {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()],
            name: match self.0 { Side::Winner => "Daily Winner Factor", Side::Loser => "Daily Loser Factor" }.into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.2.0".into(),
            tags: ["DFZQ", "price_volume", "return", "leaderboard", "exponential_decay", "daily", "neutralize", "barra", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: "Square root of 120-day exponentially decayed top/bottom-80 daily pct_chg membership, half-life 10 trading days. Multiplies the decayed sum by (1-decay^120)/(1-decay), as specified, not its reciprocal. Finite present non-BJ returns only, ties by ascending stock code; fewer than 80 eligible stocks selects all. Missing history is nonmembership, min_periods=1, invalid current returns are null. Final daily SW L1 industry and Barra SIZE regression residual; missing exposures stay null. No final zscore or sign reversal. Shared daily partial selection and bounded sparse event queues; only requested outputs computed.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["pct_chg"]).with_panel_only(),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]).with_panel_only(),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: WINDOW - 1 },
        }
    }
    fn compute_provider_key(&self) -> String {
        "dfzq_daily_leaderboard".into()
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id().into()], data)?
            .pop()
            .ok_or_else(|| err("missing leaderboard output"))
    }
    fn compute_many(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute(ids, data)
    }
}
fn compute(ids: &[String], data: &DataPool) -> Result<Vec<FactorSeries>> {
    let sides: Vec<_> = [Side::Winner, Side::Loser]
        .into_iter()
        .filter(|side| ids.iter().any(|id| id == side.id()))
        .collect();
    if sides.is_empty() {
        return Ok(vec![]);
    }
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let returns = panel.column("pct_chg")?;
    let size = panel.column_from_panel(data.daily_panel(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    columns(panel, &returns, &sides)?
        .into_iter()
        .zip(sides)
        .map(|(column, side)| {
            Ok(
                neutralize_size_sector_with_inputs(&column, panel, &size, &sector)?
                    .to_factor_series(DailyLeaderboard(side).spec()),
            )
        })
        .collect()
}
fn select(eligible: &mut [(usize, f64)], codes: &[String], side: Side) -> Vec<usize> {
    let k = TOP.min(eligible.len());
    if k < eligible.len() {
        eligible.select_nth_unstable_by(k, |a, b| {
            let order = match side {
                Side::Winner => b.1.total_cmp(&a.1),
                Side::Loser => a.1.total_cmp(&b.1),
            };
            order.then_with(|| codes[a.0].cmp(&codes[b.0]))
        });
    }
    eligible[..k].iter().map(|v| v.0).collect()
}
struct Rolling {
    sums: Vec<f64>,
    counts: Vec<usize>,
    events: VecDeque<Vec<usize>>,
}
impl Rolling {
    fn new(n: usize) -> Self {
        Self {
            sums: vec![0.0; n],
            counts: vec![0; n],
            events: VecDeque::with_capacity(WINDOW),
        }
    }
    fn step(&mut self, selected: Vec<usize>, decay: f64) {
        for value in &mut self.sums {
            *value *= decay;
        }
        if self.events.len() == WINDOW {
            let expired_weight = decay.powi(WINDOW as i32);
            for i in self.events.pop_front().unwrap() {
                self.sums[i] -= expired_weight;
                self.counts[i] -= 1;
                if self.counts[i] == 0 {
                    self.sums[i] = 0.0;
                }
            }
        }
        for &i in &selected {
            self.sums[i] += 1.0;
            self.counts[i] += 1;
        }
        self.events.push_back(selected);
    }
}
fn columns(panel: &DailyPanel, returns: &PanelColumn, sides: &[Side]) -> Result<Vec<PanelColumn>> {
    let n = panel.instruments().len();
    let decay = 0.5_f64.powf(1.0 / HALF_LIFE);
    let multiplier = (1.0 - decay.powi(WINDOW as i32)) / (1.0 - decay);
    let eligible_stock: Vec<_> = panel
        .instruments()
        .iter()
        .map(|code| !is_bj_stock(code))
        .collect();
    let mut states: Vec<_> = sides.iter().map(|_| Rolling::new(n)).collect();
    let mut outputs: Vec<_> = sides
        .iter()
        .map(|_| vec![None; panel.shape_len()])
        .collect();
    let mut eligible = Vec::with_capacity(n);
    for (day, &date) in panel.dates().iter().enumerate() {
        eligible.clear();
        for (i, &allowed) in eligible_stock.iter().enumerate() {
            let offset = day * n + i;
            if allowed && panel.is_present_offset(offset) {
                if let Some(value) = returns.values()[offset].filter(|v| v.is_finite()) {
                    // Signed zero must tie by stock code, not float bit patterns.
                    eligible.push((i, if value == 0.0 { 0.0 } else { value }));
                }
            }
        }
        for ((&side, state), values) in sides.iter().zip(&mut states).zip(&mut outputs) {
            state.step(select(&mut eligible, panel.instruments(), side), decay);
            if panel.is_target_date(date) {
                for &(i, _) in &eligible {
                    values[day * n + i] = Some((multiplier * state.sums[i].max(0.0)).sqrt());
                }
            }
        }
    }
    outputs
        .into_iter()
        .map(|v| panel.column_from_values(v))
        .collect()
}

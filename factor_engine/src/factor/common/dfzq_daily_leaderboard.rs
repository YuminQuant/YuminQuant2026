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
                DataRequest::new(DatasetId::StockDailyPv, &["pct_chg"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
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
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use std::collections::BTreeMap;
    #[test]
    fn leaderboard_final_outputs_are_neutralized_and_request_independent() {
        use std::collections::HashMap;
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260424,
            end_date: 20260424,
            load_start_date: 20260424,
            load_dates: vec![20260424],
            target_dates: vec![20260424],
        };
        let codes =
            || ColumnData::Utf8((0..100).map(|i| Some(format!("{:06}.SZ", i + 1))).collect());
        let dates = || ColumnData::I32(vec![Some(20260424); 100]);
        let pv = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("trade_date".into(), dates()),
            (
                "pct_chg".into(),
                ColumnData::F64((0..100).map(|i| Some(i as f64)).collect()),
            ),
        ]))
        .unwrap();
        let size = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("trade_date".into(), dates()),
            (
                "SIZE".into(),
                ColumnData::F64((0..100).map(|i| (i != 98).then_some(i as f64)).collect()),
            ),
        ]))
        .unwrap()
        .take(&(0..100).rev().collect::<Vec<_>>())
        .unwrap();
        let sector = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("in_date".into(), ColumnData::I32(vec![Some(20100101); 100])),
            ("out_date".into(), ColumnData::I32(vec![None; 100])),
            (
                "l1_code".into(),
                ColumnData::Utf8(
                    (0..100)
                        .map(|i| (i != 99).then(|| (i % 2).to_string()))
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockDailyPv, pv),
                (DatasetId::StockBarraDaily, size),
                (DatasetId::StockSwClassification, sector),
            ]),
            &context,
        )
        .unwrap();
        let both = compute(&["dwf".into(), "dlf".into()], &data).unwrap();
        for result in both {
            let single = compute(&[result.spec.id.clone()], &data).unwrap();
            let values: Vec<_> = result.values.iter().map(|v| v.value).collect();
            assert_eq!(
                values,
                single[0].values.iter().map(|v| v.value).collect::<Vec<_>>()
            );
            assert_eq!(&values[98..], &[None, None]);
            assert!(values[..98].iter().all(Option::is_some));
            assert!(values[..98].iter().flatten().any(|v| *v < 0.0));
            for sector in 0..2 {
                assert!(
                    (sector..98)
                        .step_by(2)
                        .map(|i| values[i].unwrap())
                        .sum::<f64>()
                        .abs()
                        < 1e-8
                );
            }
            assert!(
                (0..98)
                    .map(|i| i as f64 * values[i].unwrap())
                    .sum::<f64>()
                    .abs()
                    < 1e-8
            );
        }
    }
    #[test]
    fn leaderboard_selects_exactly_80_and_breaks_ties_by_code() {
        let codes: Vec<_> = (0..100).map(|i| format!("{i:06}.SZ")).collect();
        let mut rows: Vec<_> = (0..100).rev().map(|i| (i, 1.0)).collect();
        for side in [Side::Winner, Side::Loser] {
            let mut picked = select(&mut rows, &codes, side);
            picked.sort_unstable();
            assert_eq!(picked, (0..80).collect::<Vec<_>>());
        }
        let mut rows: Vec<_> = (0..100).map(|i| (i, i as f64)).collect();
        assert!(select(&mut rows, &codes, Side::Winner)
            .iter()
            .all(|i| *i >= 20));
        assert!(select(&mut rows, &codes, Side::Loser)
            .iter()
            .all(|i| *i < 80));
        assert_eq!(select(&mut rows[..3], &codes, Side::Winner).len(), 3);
        assert!(select(&mut [], &codes, Side::Loser).is_empty());
    }
    #[test]
    fn leaderboard_recurrence_matches_finite_window_and_half_life() {
        let d = 0.5_f64.powf(1.0 / HALF_LIFE);
        assert!((d.powi(10) - 0.5).abs() < 1e-14);
        let mut rolling = Rolling::new(1);
        for t in 0..500 {
            rolling.step(if t % 7 == 0 { vec![0] } else { vec![] }, d);
            let expected: f64 = (0..=t)
                .filter(|s| t - s < WINDOW && s % 7 == 0)
                .map(|s| d.powi((t - s) as i32))
                .sum();
            assert!((rolling.sums[0] - expected).abs() < 1e-12);
            assert!(rolling.events.len() <= WINDOW);
        }
        let mut isolated = Rolling::new(1);
        isolated.step(vec![0], d);
        for _ in 0..WINDOW {
            isolated.step(vec![], d);
        }
        assert_eq!(isolated.sums[0], 0.0);
    }
    #[test]
    fn leaderboard_panel_masks_scale_and_requested_outputs() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260102,
            end_date: 20260102,
            load_start_date: 20260101,
            load_dates: vec![20260101, 20260102],
            target_dates: vec![20260102],
        };
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".into(),
                ColumnData::I32(
                    vec![Some(20260101); 3]
                        .into_iter()
                        .chain(vec![Some(20260102); 3])
                        .collect(),
                ),
            ),
            (
                "ts_code".into(),
                ColumnData::Utf8(
                    [
                        "000001.SZ",
                        "000002.SZ",
                        "430001.BJ",
                        "000001.SZ",
                        "000002.SZ",
                        "430001.BJ",
                    ]
                    .into_iter()
                    .map(|s| Some(s.into()))
                    .collect(),
                ),
            ),
            (
                "pct_chg".into(),
                ColumnData::F64(vec![
                    Some(1.0),
                    Some(2.0),
                    Some(99.0),
                    Some(2.0),
                    None,
                    Some(99.0),
                ]),
            ),
        ]))
        .unwrap();
        let panel = DailyPanel::from_table(&table, &context).unwrap();
        let ret = panel.column("pct_chg").unwrap();
        let both = columns(&panel, &ret, &[Side::Winner, Side::Loser]).unwrap();
        let one = columns(&panel, &ret, &[Side::Loser]).unwrap();
        assert_eq!(both[1].values(), one[0].values());
        let d = 0.5_f64.powf(1.0 / HALF_LIFE);
        let expected = ((1.0 - d.powi(120)) / (1.0 - d) * (1.0 + d)).sqrt();
        assert!((both[0].values()[3].unwrap() - expected).abs() < 1e-12);
        assert_eq!(both[0].values()[4], None);
        assert_eq!(both[0].values()[5], None);
        assert!(both[0].values()[..3].iter().all(Option::is_none));
        for side in [Side::Winner, Side::Loser] {
            let spec = DailyLeaderboard(side).spec();
            assert!(spec.tags.contains(&"price_volume".into()));
            assert!(spec.tags.contains(&"DFZQ".into()));
            assert!(spec.tags.contains(&"neutralize".into()));
            assert_eq!(spec.dependencies.len(), 3);
            assert_eq!(spec.lookback.trading_days, 119);
        }
        assert_eq!(
            DailyLeaderboard(Side::Winner).compute_provider_key(),
            DailyLeaderboard(Side::Loser).compute_provider_key()
        );
    }
}

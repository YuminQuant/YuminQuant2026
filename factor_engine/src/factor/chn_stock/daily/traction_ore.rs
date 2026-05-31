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
const WINDOW: usize = 80;
const LOOKBACK_DAYS: usize = WINDOW + 1;
const MIN_PAIR_DAYS: u8 = 40;
const PRUNE_FRACTION: f64 = 0.20;

pub struct StockDailyTractionOre;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTractionOre)
}

impl Factor for StockDailyTractionOre {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "traction_ore".to_string(),
            aliases: vec!["Traction_ORE".to_string()],
            name: "traction_ore".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ overnight-return purified cross-sectional network traction factor. It builds an 80-day overnight-return cosine network after removing gap-reversal samples, prunes the weakest 20% edges, computes association-weighted peer Ret20 ExpAve, and neutralizes by Ret20, Barra SIZE, and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["open", "high", "low", "pre_close", "close"],
                ),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: LOOKBACK_DAYS,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let overnight = purified_overnight_returns(&panel, data)?;
        let ret20 = adjusted_20d_return(data, &panel)?;
        let expave = traction_ore_expave(&overnight, &ret20, &panel)?;
        let factor = neutralize_ret20_size_sector(&expave, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "cs_network",
        "overnight",
        "gap_purified",
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

fn purified_overnight_returns(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let pv = data.daily(DatasetId::StockDailyPv)?;
    let open = panel.column_from_table(pv, "open")?;
    let high = panel.column_from_table(pv, "high")?;
    let low = panel.column_from_table(pv, "low")?;
    let pre_close = panel.column_from_table(pv, "pre_close")?;
    let code_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let eligible = eligible_instruments(panel);
    let mut output = vec![None; panel.shape_len()];

    for date_idx in 0..date_count {
        let offset = date_idx * code_count;
        for code_idx in 0..code_count {
            if !eligible[code_idx] {
                continue;
            }
            let sample = overnight_sample(
                open.values()[offset + code_idx],
                pre_close.values()[offset + code_idx],
            );
            if date_idx < 2 {
                continue;
            }
            let previous_offset = (date_idx - 1) * code_count;
            let before_previous_offset = (date_idx - 2) * code_count;
            let previous_gap = gap_kind(
                open.values()[previous_offset + code_idx],
                high.values()[before_previous_offset + code_idx],
                low.values()[before_previous_offset + code_idx],
            );
            output[offset + code_idx] = purify_overnight_sample(sample, previous_gap);
        }
    }

    panel.column_from_values(output)
}

fn overnight_sample(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    let (Some(open), Some(pre_close)) = (clean(open), clean(pre_close)) else {
        return None;
    };
    if pre_close <= f64::EPSILON {
        return None;
    }
    let value = open / pre_close - 1.0;
    value.is_finite().then_some(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GapKind {
    None,
    Up,
    Down,
    Invalid,
}

fn gap_kind(open: Option<f64>, previous_high: Option<f64>, previous_low: Option<f64>) -> GapKind {
    let (Some(open), Some(previous_high), Some(previous_low)) =
        (clean(open), clean(previous_high), clean(previous_low))
    else {
        return GapKind::Invalid;
    };
    if open > previous_high {
        GapKind::Up
    } else if open < previous_low {
        GapKind::Down
    } else {
        GapKind::None
    }
}

fn purify_overnight_sample(sample: Option<f64>, previous_gap: GapKind) -> Option<f64> {
    let sample = clean(sample)?;
    match previous_gap {
        GapKind::Up if sample < 0.0 => None,
        GapKind::Down if sample > 0.0 => None,
        GapKind::Invalid => None,
        _ => Some(sample),
    }
}

fn traction_ore_expave(
    overnight: &PanelColumn,
    associated_returns: &PanelColumn,
    panel: &DailyPanel,
) -> Result<PanelColumn> {
    let code_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let eligible = eligible_instruments(panel);
    let mut output = vec![None; panel.shape_len()];
    let mut state = PairCosineState::new(code_count);

    for date_idx in 0..date_count {
        if date_idx >= WINDOW {
            let remove_offset = (date_idx - WINDOW) * code_count;
            state.remove_day(
                &overnight.values()[remove_offset..remove_offset + code_count],
                &eligible,
            );
        }

        let add_offset = date_idx * code_count;
        state.add_day(
            &overnight.values()[add_offset..add_offset + code_count],
            &eligible,
        );

        if date_idx < LOOKBACK_DAYS {
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

struct PairCosineState {
    code_count: usize,
    dot: Vec<f64>,
    left_sq: Vec<f64>,
    right_sq: Vec<f64>,
    valid_count: Vec<u8>,
}

impl PairCosineState {
    fn new(code_count: usize) -> Self {
        let pair_count = code_count.saturating_mul(code_count.saturating_sub(1)) / 2;
        Self {
            code_count,
            dot: vec![0.0; pair_count],
            left_sq: vec![0.0; pair_count],
            right_sq: vec![0.0; pair_count],
            valid_count: vec![0; pair_count],
        }
    }

    fn add_day(&mut self, samples: &[Option<f64>], eligible: &[bool]) {
        self.update_day(samples, eligible, 1.0, 1);
    }

    fn remove_day(&mut self, samples: &[Option<f64>], eligible: &[bool]) {
        self.update_day(samples, eligible, -1.0, -1);
    }

    fn update_day(
        &mut self,
        samples: &[Option<f64>],
        eligible: &[bool],
        value_delta: f64,
        count_delta: i8,
    ) {
        if self.code_count < 2 {
            return;
        }
        for left in 0..self.code_count - 1 {
            if !eligible[left] {
                continue;
            }
            let Some(left_value) = clean(samples[left]) else {
                continue;
            };
            let left_square = left_value * left_value;
            let mut pair_idx = pair_index(left, left + 1, self.code_count);
            for right in left + 1..self.code_count {
                if eligible[right] {
                    if let Some(right_value) = clean(samples[right]) {
                        update_float(
                            &mut self.dot[pair_idx],
                            value_delta * left_value * right_value,
                        );
                        update_float(&mut self.left_sq[pair_idx], value_delta * left_square);
                        update_float(
                            &mut self.right_sq[pair_idx],
                            value_delta * right_value * right_value,
                        );
                        update_counter(&mut self.valid_count[pair_idx], count_delta);
                    }
                }
                pair_idx += 1;
            }
        }
    }

    fn edge_weight(&self, pair_idx: usize) -> Option<f64> {
        if self.valid_count[pair_idx] < MIN_PAIR_DAYS {
            return None;
        }
        let left_sq = self.left_sq[pair_idx];
        let right_sq = self.right_sq[pair_idx];
        if left_sq <= f64::EPSILON || right_sq <= f64::EPSILON {
            return None;
        }
        let cosine = self.dot[pair_idx] / (left_sq * right_sq).sqrt();
        if !cosine.is_finite() {
            return None;
        }
        Some(((cosine.clamp(-1.0, 1.0)) + 1.0) * 0.5)
    }

    fn expave(&self, associated_returns: &[Option<f64>], eligible: &[bool]) -> Vec<Option<f64>> {
        let mut weights = Vec::new();
        if self.code_count >= 2 {
            for left in 0..self.code_count - 1 {
                if !eligible[left] {
                    continue;
                }
                let mut pair_idx = pair_index(left, left + 1, self.code_count);
                for right in left + 1..self.code_count {
                    if eligible[right] {
                        if let Some(weight) = self.edge_weight(pair_idx) {
                            weights.push(weight);
                        }
                    }
                    pair_idx += 1;
                }
            }
        }

        let Some(threshold) = edge_prune_threshold(&mut weights) else {
            return vec![None; self.code_count];
        };

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
                        if let Some(weight) = self.edge_weight(pair_idx) {
                            if weight >= threshold {
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

fn edge_prune_threshold(weights: &mut [f64]) -> Option<f64> {
    if weights.is_empty() {
        return None;
    }
    let prune_count = ((weights.len() as f64) * PRUNE_FRACTION).floor() as usize;
    if prune_count == 0 {
        return Some(f64::NEG_INFINITY);
    }
    let threshold_idx = prune_count.min(weights.len() - 1);
    let (_, threshold, _) =
        weights.select_nth_unstable_by(threshold_idx, |left, right| left.total_cmp(right));
    Some(*threshold)
}

fn update_float(value: &mut f64, delta: f64) {
    *value += delta;
    if value.abs() < 1e-12 {
        *value = 0.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn traction_ore_overnight_return_uses_open_over_preclose() {
        assert_close(overnight_sample(Some(11.0), Some(10.0)), 0.1);
        assert_eq!(overnight_sample(Some(11.0), Some(0.0)), None);
        assert_eq!(overnight_sample(Some(f64::NAN), Some(10.0)), None);
    }

    #[test]
    fn traction_ore_gap_kind_uses_unadjusted_open_against_prior_range() {
        assert_eq!(gap_kind(Some(12.0), Some(11.0), Some(9.0)), GapKind::Up);
        assert_eq!(gap_kind(Some(8.0), Some(11.0), Some(9.0)), GapKind::Down);
        assert_eq!(gap_kind(Some(10.0), Some(11.0), Some(9.0)), GapKind::None);
        assert_eq!(gap_kind(None, Some(11.0), Some(9.0)), GapKind::Invalid);
    }

    #[test]
    fn traction_ore_gap_reversal_purification_filters_samples() {
        assert_eq!(purify_overnight_sample(Some(-0.01), GapKind::Up), None);
        assert_eq!(purify_overnight_sample(Some(0.01), GapKind::Down), None);
        assert_close(purify_overnight_sample(Some(0.01), GapKind::Up), 0.01);
        assert_close(purify_overnight_sample(Some(-0.01), GapKind::Down), -0.01);
        assert_eq!(purify_overnight_sample(Some(0.01), GapKind::Invalid), None);
    }

    #[test]
    fn traction_ore_pair_index_is_lower_triangle_without_self() {
        assert_eq!(pair_index(0, 1, 4), 0);
        assert_eq!(pair_index(0, 2, 4), 1);
        assert_eq!(pair_index(0, 3, 4), 2);
        assert_eq!(pair_index(1, 2, 4), 3);
        assert_eq!(pair_index(1, 3, 4), 4);
        assert_eq!(pair_index(2, 3, 4), 5);
    }

    #[test]
    fn traction_ore_requires_minimum_common_valid_pair_days() {
        let eligible = vec![true, true];
        let mut state = PairCosineState::new(2);
        for _ in 0..39 {
            state.add_day(&[Some(1.0), Some(1.0)], &eligible);
        }
        assert_eq!(
            state.expave(&[Some(0.1), Some(0.2)], &eligible),
            vec![None, None]
        );

        state.add_day(&[Some(1.0), Some(1.0)], &eligible);
        let expave = state.expave(&[Some(0.1), Some(0.2)], &eligible);
        assert_close(expave[0], 0.2);
        assert_close(expave[1], 0.1);
    }

    #[test]
    fn traction_ore_rolling_remove_updates_pair_state() {
        let eligible = vec![true, true];
        let mut state = PairCosineState::new(2);
        for _ in 0..40 {
            state.add_day(&[Some(1.0), Some(2.0)], &eligible);
        }
        assert_eq!(state.valid_count[0], 40);
        state.remove_day(&[Some(1.0), Some(2.0)], &eligible);
        assert_eq!(state.valid_count[0], 39);
        assert_eq!(
            state.expave(&[Some(0.1), Some(0.2)], &eligible),
            vec![None, None]
        );
    }

    #[test]
    fn traction_ore_cosine_weight_maps_to_zero_one_range() {
        let eligible = vec![true, true];
        let mut state = PairCosineState::new(2);
        for _ in 0..40 {
            state.add_day(&[Some(1.0), Some(-1.0)], &eligible);
        }
        assert_close(state.edge_weight(0), 0.0);

        let mut state = PairCosineState::new(2);
        for _ in 0..40 {
            state.add_day(&[Some(2.0), Some(4.0)], &eligible);
        }
        assert_close(state.edge_weight(0), 1.0);
    }

    #[test]
    fn traction_ore_edge_prune_threshold_drops_lowest_twenty_percent_rank() {
        let mut weights = vec![0.9, 0.1, 0.7, 0.2, 0.5, 0.4, 0.8, 0.3, 0.6, 1.0];
        assert_close(edge_prune_threshold(&mut weights), 0.3);
    }

    #[test]
    fn traction_ore_bj_is_not_eligible_for_network_output() {
        let eligible = vec![true, false, true];
        let mut state = PairCosineState::new(3);
        for _ in 0..40 {
            state.add_day(&[Some(1.0), Some(1.0), Some(1.0)], &eligible);
        }

        let expave = state.expave(&[Some(0.1), Some(0.2), Some(0.4)], &eligible);

        assert_close(expave[0], 0.4);
        assert_eq!(expave[1], None);
        assert_close(expave[2], 0.1);
    }

    #[test]
    fn traction_ore_spec_has_kyzq_cs_network_and_lookback_tags() {
        let spec = StockDailyTractionOre.spec();
        assert_eq!(spec.id, "traction_ore");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "cs_network"));
        assert!(spec.tags.iter().any(|tag| tag == "gap_purified"));
        assert_eq!(spec.lookback.trading_days, LOOKBACK_DAYS);
    }

    #[test]
    fn traction_ore_source_has_no_inner_parallelism_keywords() {
        let source = include_str!("traction_ore.rs");
        let needles = [
            ['r', 'a', 'y', 'o', 'n'].iter().collect::<String>(),
            ['p', 'a', 'r', '_', 'i', 't', 'e', 'r']
                .iter()
                .collect::<String>(),
            [
                'i', 'n', 't', 'o', '_', 'p', 'a', 'r', '_', 'i', 't', 'e', 'r',
            ]
            .iter()
            .collect::<String>(),
        ];
        for needle in needles {
            assert!(!source.contains(&needle));
        }
    }
}

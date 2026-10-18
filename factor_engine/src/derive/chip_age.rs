//! Causal chip-age estimation. Engineering choices are recorded in manifest.json.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::parquet_io::read_parquet;
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};

const GRID: f64 = 0.01;
type Distribution = BTreeMap<i32, f64>;

const WINDOW: usize = 250;

struct ChipState {
    // Raw surviving masses, NOT normalized between days. Each cohort expires
    // individually at age 251; normalizing the state would reintroduce history.
    ages: VecDeque<Distribution>,
    adjusted_close: f64,
}

impl ChipState {
    fn new(adjusted_close: f64) -> Self {
        Self { ages: VecDeque::new(), adjusted_close }
    }

    fn advance(&mut self, turnover: f64, mut incoming: Distribution, close: f64) -> Result<()> {
        let survive = (-turnover).exp();
        let replace = -(-turnover).exp_m1();
        for distribution in &mut self.ages {
            for value in distribution.values_mut() { *value *= survive; }
        }
        for value in incoming.values_mut() { *value *= replace; }
        self.ages.push_front(incoming);
        if self.ages.len() > WINDOW { self.ages.pop_back(); }
        self.adjusted_close = close;
        Ok(())
    }

    fn retained_mass(&self) -> f64 {
        self.ages.iter().flat_map(|d| d.values()).sum()
    }

    fn features(&self) -> Option<([f32; 128], f64)> {
        if self.ages.len() != WINDOW || !self.adjusted_close.is_finite() || self.adjusted_close <= 0.0 { return None; }
        let mut masses = [0.0_f64; 128];
        let mut outside = 0.0;
        for (index, distribution) in self.ages.iter().enumerate() {
            let group = match index { 0..=1 => 0, 2..=9 => 1, 10..=99 => 2, _ => 3 };
            for (&bin, &mass) in distribution {
                let relative = (bin as f64 + 0.5) * GRID - self.adjusted_close.ln();
                if !(-0.7..=0.7).contains(&relative) { outside += mass; }
                let target = (((relative + 0.7) / 1.4 * 32.0).floor() as i32).clamp(0, 31) as usize;
                masses[group * 32 + target] += mass;
            }
        }
        let total: f64 = masses.iter().sum();
        if !total.is_finite() || total <= 0.0 { return None; }
        let mut result = [0.0_f32; 128];
        for (out, mass) in result.iter_mut().zip(masses) { *out = (mass / total).sqrt() as f32; }
        Some((result, outside / total))
    }
}

fn triangle(low: f64, high: f64, mode: f64) -> Result<Distribution> {
    if !low.is_finite() || !high.is_finite() || !mode.is_finite()
        || low <= 0.0 || high < low {
        return Err(err("invalid triangular chip price distribution"));
    }
    let first = (low.ln() / GRID).floor() as i32;
    let last = (high.ln() / GRID).floor() as i32;
    if first == last || high == low { return Ok(BTreeMap::from([(first, 1.0)])); }
    let cdf = |x: f64| -> f64 {
        if x <= low { 0.0 } else if x >= high { 1.0 }
        else if x <= mode && mode > low { (x - low).powi(2) / ((high - low) * (mode - low)) }
        else { 1.0 - (high - x).powi(2) / ((high - low) * (high - mode)) }
    };
    let mut result = BTreeMap::new();
    for bin in first..=last {
        let mass = cdf(((bin + 1) as f64 * GRID).exp()) - cdf((bin as f64 * GRID).exp());
        if mass > 0.0 { result.insert(bin, mass); }
    }
    let total: f64 = result.values().sum();
    if total <= 0.0 { return Err(err("empty triangular chip distribution")); }
    for mass in result.values_mut() { *mass /= total; }
    Ok(result)
}

fn numeric_rows(table: &Table, columns: &[&str]) -> Result<BTreeMap<String, Vec<Option<f64>>>> {
    let codes = table.required_utf8("ts_code")?;
    let values = columns.iter().map(|c| table.required_f64_cast(c)).collect::<Result<Vec<_>>>()?;
    let mut rows = BTreeMap::new();
    for (i, code) in codes.iter().enumerate() {
        if let Some(code) = code {
            if rows.insert(code.clone(), values.iter().map(|v| v[i]).collect()).is_some() {
                return Err(err(format!("duplicate source symbol: {code}")));
            }
        }
    }
    Ok(rows)
}

fn full_day_suspensions(root: &Path, date: i32) -> Result<BTreeSet<String>> {
    let path = root.join("chn_stock_data/daily/suspend").join((date / 10000).to_string()).join(format!("{date}.parquet"));
    if !path.exists() { return Ok(BTreeSet::new()); }
    let table = read_parquet(&path, None)?;
    let codes = table.required_utf8("ts_code")?;
    let types = table.required_utf8("suspend_type")?;
    let timing = table.required_utf8("suspend_timing")?;
    Ok((0..table.len).filter(|&i| types[i].as_deref() == Some("S")
        && timing[i].as_deref().unwrap_or("").trim().is_empty())
        .filter_map(|i| codes[i].clone()).collect())
}

pub fn derive_chip_age(config: &EngineConfig, start: i32, end: i32) -> Result<()> {
    if start > end { return Err(err("require start-date <= end-date")); }
    // No system temporary directory or hidden preprocessing cache is used.
    let root = config.data_root.join("derived/stock/chip_age_rolling250_v2");
    let absolute = if root.is_absolute() { root.clone() } else { std::env::current_dir()?.join(&root) };
    if absolute.to_string_lossy().to_ascii_lowercase().starts_with("c:") {
        return Err(err("chip-age output on C: is prohibited; configure data_root on D:"));
    }
    std::fs::create_dir_all(&root)?;
    let manifest = root.join("manifest.json");
    // Remove the readiness marker before changing any daily file. A failed run
    // cannot be consumed as a completed dataset by the Python provider.
    if manifest.exists() { std::fs::remove_file(&manifest)?; }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let targets = calendar.open_dates_between(start, end);
    let first = *targets.first().ok_or_else(|| err("no chip-age target trading dates"))?;
    let dates = calendar.open_dates_between(calendar.warmup_start(first, WINDOW - 1), end);
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root.clone()));
    let mut states: BTreeMap<String, ChipState> = BTreeMap::new();
    for date in dates {
        let read = |dataset, names: &[&str]| -> Result<BTreeMap<String, Vec<Option<f64>>>> {
            let cols = names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            numeric_rows(&loader.load_daily_by_dates(dataset, &cols, &[date])?, names)
        };
        let prices = read(DatasetId::StockDailyPv, &["low", "high", "close", "amount", "vol"])?;
        let turnover = read(DatasetId::StockDailyBasic, &["turnover_rate_f"])?;
        let adjustment = read(DatasetId::StockAdjFactor, &["adj_factor"])?;
        let suspensions = full_day_suspensions(&config.data_root, date)?;
        let symbols: BTreeSet<_> = states.keys().chain(prices.keys()).chain(suspensions.iter()).cloned().collect();
        let mut output = Vec::new();
        let mut invalid = 0;
        let mut invalid_symbols = Vec::new();
        let mut high_turnover = 0;
        for code in symbols {
            let raw_volume = prices.get(&code).and_then(|v| v[4]);
            if suspensions.contains(&code) && raw_volume.map_or(true, |v| v == 0.0) {
                // A suspension at the left edge still counts as an observed
                // zero-turnover session. It must not shorten a replay's window.
                let state = states.entry(code.clone()).or_insert_with(|| ChipState::new(f64::NAN));
                state.advance(0.0, Distribution::new(), state.adjusted_close)?;
                if let Some((features, outside)) = state.features() {
                    output.push((code, features, outside, state.retained_mass(), state.ages.len()));
                }
                continue;
            }
            let source = prices.get(&code).and_then(|v| {
                let adj = adjustment.get(&code)?[0]?;
                let rate = turnover.get(&code)?[0]? / 100.0;
                let (low, high, close, amount, volume) = (v[0]?, v[1]?, v[2]?, v[3]?, v[4]?);
                if ![adj, rate, low, high, close, amount, volume].iter().all(|v| v.is_finite())
                    || adj <= 0.0 || rate < 0.0 || low <= 0.0 || high < low
                    || close <= 0.0 || volume <= 0.0 || amount <= 0.0 { return None; }
                Some((low * adj, high * adj, close * adj, amount * 10.0 / volume * adj, rate))
            });
            let Some((low, high, close, vwap, rate)) = source else {
                states.remove(&code);
                if invalid_symbols.len() < 20 { invalid_symbols.push(code.clone()); }
                invalid += 1;
                continue;
            };
            let distribution = triangle(low, high, vwap)?;
            if rate > 1.0 { high_turnover += 1; }
            states.entry(code.clone()).or_insert_with(|| ChipState::new(close))
                .advance(rate, distribution, close)?;
            let state = &states[&code];
            if let Some((features, outside)) = state.features() {
                output.push((code, features, outside, state.retained_mass(), state.ages.len()));
            }
        }
        let count = output.len();
        if date >= start {
            let mut columns = BTreeMap::new();
            columns.insert("trade_date".into(), ColumnData::I32(vec![Some(date); count]));
            columns.insert("ts_code".into(), ColumnData::Utf8(output.iter().map(|r| Some(r.0.clone())).collect()));
            for i in 0..128 { columns.insert(format!("chip_age_{}_{:02}", i / 32, i % 32),
                ColumnData::F32(output.iter().map(|r| Some(r.1[i])).collect())); }
            columns.insert("outside_mass".into(), ColumnData::F64(output.iter().map(|r| Some(r.2)).collect()));
            columns.insert("retained_window_mass".into(), ColumnData::F64(output.iter().map(|r| Some(r.3)).collect()));
            columns.insert("history_days".into(), ColumnData::I32(output.iter().map(|r| Some(r.4 as i32)).collect()));
            let table = Table::new(columns)?;
            let directory = root.join((date / 10000).to_string());
            std::fs::create_dir_all(&directory)?;
            let path = directory.join(format!("{date}.parquet"));
            let temp = path.with_extension(format!("{}.tmp", std::process::id()));
            write_derived_parquet(&temp, &table)?;
            std::fs::rename(temp, path)?;
        }
        println!("derive-chip-age date={date} rows={count} states={} invalid_reset={invalid} invalid_first20={invalid_symbols:?} turnover_above_100pct={high_turnover} warmup={}", states.len(), date < start);
    }
    // Definition identity stays stable when new OOS dates are appended.
    let body = r#"{"version":"chip_age_rolling250_v2","window_trading_days":250,
        "turnover":"turnover_rate_f","replacement":"1-exp(-rate/100)",
        "grid_log_step":0.01,"initialization":"zero_mass",
        "normalization":"output_only","age_groups":[[1,2],[3,10],[11,100],[101,250]],
        "overflow":"edge_bins","invalid_data":"reset_and_refill_window",
        "suspension":"explicit_full_day_carry_adjusted_close"}"#;
    let temp = manifest.with_extension("json.tmp");
    std::fs::write(&temp, body)?;
    std::fs::rename(temp, manifest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_mass_and_endpoint_modes() {
        for mode in [10.0, 11.0, 12.0] {
            let distribution = triangle(10.0, 12.0, mode).unwrap();
            assert!((distribution.values().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(distribution.values().all(|v| *v >= 0.0));
        }
        assert_eq!(triangle(10.0, 10.0, 10.0).unwrap().values().sum::<f64>(), 1.0);
    }

    #[test]
    fn first_cohort_is_not_initialized_to_full_mass() {
        let mut state = ChipState::new(1.0);
        state.advance(1.5, BTreeMap::from([(0, 1.0)]), 1.0).unwrap();
        assert!((state.retained_mass() - (1.0 - (-1.5_f64).exp())).abs() < 1e-12);
        for _ in 1..WINDOW { state.advance(0.0, Distribution::new(), 1.0).unwrap(); }
        assert!(state.features().is_some());
        state.advance(0.0, Distribution::new(), 1.0).unwrap();
        assert_eq!(state.retained_mass(), 0.0);
        assert!(state.features().is_none());
    }

    #[test]
    fn rolling_state_matches_direct_window_formula_and_is_origin_independent() {
        let mut long = ChipState::new(1.0);
        for _ in 0..87 { long.advance(0.4, BTreeMap::from([(-99, 1.0)]), 1.0).unwrap(); }
        let mut short = ChipState::new(1.0);
        let rates: Vec<f64> = (0..WINDOW).map(|i| 0.001 * (1 + i % 7) as f64).collect();
        for (i, &rate) in rates.iter().enumerate() {
            let distribution = BTreeMap::from([(i as i32, 1.0)]);
            long.advance(rate, distribution.clone(), 1.0).unwrap();
            short.advance(rate, distribution, 1.0).unwrap();
        }
        assert_eq!(long.ages, short.ages);
        for (i, &rate) in rates.iter().enumerate() {
            let expected = -(-rate).exp_m1() * (-rates[i + 1..].iter().sum::<f64>()).exp();
            assert!((short.ages[WINDOW - 1 - i][&(i as i32)] - expected).abs() < 1e-12);
        }
        let (x, _) = short.features().unwrap();
        assert!((x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn complete_250_day_window_is_required() {
        let mut state = ChipState::new(1.0);
        for _ in 0..249 { state.advance(0.001, BTreeMap::from([(0, 1.0)]), 1.0).unwrap(); }
        assert!(state.features().is_none());
        state.advance(0.001, BTreeMap::from([(0, 1.0)]), 1.0).unwrap();
        assert!(state.features().is_some());
    }
}

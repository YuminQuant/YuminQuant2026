use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::backtest::request::BacktestRunRequest;
use crate::barra::engine::DEFAULT_BARRA_MODEL;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, Frequency};
use crate::data::parquet_io::{parquet_column_names, read_parquet};
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::storage::{FactorMetadata, FactorStorage, LabelStorage};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FactorRootLayout {
    Standard,
    DirectDaily,
}

#[derive(Clone, Debug)]
pub struct BacktestInput {
    pub factor_metadata: Vec<FactorMetadata>,
    pub label_metadata: LabelMetadataInfo,
    pub target_dates: Vec<i32>,
    pub all_dates: Vec<i32>,
    pub panel: BacktestPanel,
    pub universe: BacktestUniverseBatch,
    pub trade_filter: BacktestTradeFilterBatch,
    pub benchmark: BenchmarkBatch,
    pub sectors: Option<HashMap<i32, Vec<Option<String>>>>,
}

#[derive(Clone, Debug)]
pub struct BacktestDataPlan {
    pub factor_metadata: Vec<FactorMetadata>,
    pub label_metadata: LabelMetadataInfo,
    pub target_dates: Vec<i32>,
    pub all_dates: Vec<i32>,
    pub instruments: Vec<String>,
    label_table: Table,
    barra_columns: Vec<String>,
    barra_table: Option<Table>,
    sector_map: Option<ClassificationMap>,
    universe: BacktestUniversePlan,
    trade_filter: BacktestTradeFilterPlan,
    benchmark: BenchmarkPlan,
}

#[derive(Clone, Debug)]
pub struct FactorFillState {
    latest: BTreeMap<String, Vec<Option<f64>>>,
    initialized: bool,
}

impl FactorFillState {
    pub fn new(factor_columns: &[String], instrument_count: usize) -> Self {
        let latest = factor_columns
            .iter()
            .map(|column| (column.clone(), vec![None; instrument_count]))
            .collect();
        Self {
            latest,
            initialized: false,
        }
    }
}

#[derive(Clone, Debug)]
struct FactorLoadResult {
    table: Table,
    present_dates: BTreeMap<String, BTreeSet<i32>>,
}

#[derive(Clone, Debug)]
pub struct BacktestUniversePlan {
    pub id: String,
    masks: HashMap<i32, Vec<bool>>,
}

#[derive(Clone, Debug)]
pub struct BacktestUniverseBatch {
    pub id: String,
    pub masks: HashMap<i32, Vec<bool>>,
}

impl BacktestUniverseBatch {
    pub fn mask_for(&self, date: i32) -> Option<&[bool]> {
        self.masks.get(&date).map(Vec::as_slice)
    }
}

#[derive(Clone, Debug)]
pub struct BacktestTradeFilterPlan {
    masks: HashMap<i32, Vec<bool>>,
}

#[derive(Clone, Debug)]
pub struct BacktestTradeFilterBatch {
    masks: HashMap<i32, Vec<bool>>,
}

impl BacktestTradeFilterBatch {
    pub fn mask_for(&self, date: i32) -> Option<&[bool]> {
        self.masks.get(&date).map(Vec::as_slice)
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkPlan {
    pub id: String,
    kind: BenchmarkKind,
}

#[derive(Clone, Debug)]
pub struct BenchmarkBatch {
    pub id: String,
    pub kind: BenchmarkKind,
}

#[derive(Clone, Debug)]
pub enum BenchmarkKind {
    MarketMean,
    Weighted(HashMap<i32, Vec<Option<f64>>>),
}

#[derive(Clone, Debug)]
pub struct LabelMetadataInfo {
    pub label_id: String,
    pub output_column: String,
    pub lookahead: usize,
}

#[derive(Clone, Debug)]
pub struct BacktestPanel {
    dates: Vec<i32>,
    instruments: Vec<String>,
    date_lookup: BTreeMap<i32, usize>,
    columns: BTreeMap<String, Vec<Option<f64>>>,
    presence: BTreeMap<String, Vec<bool>>,
}

impl BacktestPanel {
    pub fn dates(&self) -> &[i32] {
        &self.dates
    }

    pub fn instruments(&self) -> &[String] {
        &self.instruments
    }

    pub fn date_index(&self, date: i32) -> Option<usize> {
        self.date_lookup.get(&date).copied()
    }

    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| err(format!("backtest panel missing column {name}")))
    }

    pub fn cross_section(&self, name: &str, date_idx: usize) -> Result<Vec<Option<f64>>> {
        let column = self.column(name)?;
        let start = date_idx * self.instruments.len();
        let end = start + self.instruments.len();
        Ok(column[start..end].to_vec())
    }

    pub fn cross_section_presence(&self, name: &str, date_idx: usize) -> Result<Vec<bool>> {
        let column = self
            .presence
            .get(name)
            .ok_or_else(|| err(format!("backtest panel missing presence mask for {name}")))?;
        let start = date_idx * self.instruments.len();
        let end = start + self.instruments.len();
        Ok(column[start..end].to_vec())
    }

    fn ensure_columns(&mut self, names: &[String]) {
        let shape_len = self.dates.len() * self.instruments.len();
        for name in names {
            self.columns
                .entry(name.clone())
                .or_insert_with(|| vec![None; shape_len]);
            self.presence
                .entry(name.clone())
                .or_insert_with(|| vec![false; shape_len]);
        }
    }
}

pub fn load_backtest_input(
    config: &EngineConfig,
    request: &BacktestRunRequest,
) -> Result<BacktestInput> {
    let plan = prepare_backtest_data_plan(config, request)?;
    let factor_columns = plan
        .factor_metadata
        .iter()
        .map(|row| row.output_column.clone())
        .collect::<Vec<_>>();
    let mut fill_state = FactorFillState::new(&factor_columns, plan.instruments.len());
    load_backtest_input_batch(
        config,
        request,
        &plan,
        &plan.factor_metadata,
        &plan.target_dates,
        &mut fill_state,
    )
}

pub fn prepare_backtest_data_plan(
    config: &EngineConfig,
    request: &BacktestRunRequest,
) -> Result<BacktestDataPlan> {
    if request.asset_class != AssetClass::Stock || request.frequency != Frequency::Daily {
        return Err(err(
            "backtest v1 only supports --asset stock --frequency daily",
        ));
    }

    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let effective_start = request
        .start_date
        .max(universe_list_date_floor(&request.universe).unwrap_or(request.start_date));
    let target_dates = calendar.open_dates_between(effective_start, request.end_date);
    if target_dates.is_empty() {
        return Err(err("no trading dates in requested backtest range"));
    }
    let factor_metadata = select_factors(config, request)?;
    if factor_metadata.is_empty() {
        return Err(err("no factors selected for backtest"));
    }
    let label_metadata = select_label(config, &request.label_id)?;
    let label_end = target_dates
        .last()
        .and_then(|date| calendar.open_date_after(*date, label_metadata.lookahead))
        .unwrap_or(*target_dates.last().expect("non-empty target dates"));
    let all_dates = calendar.open_dates_between(effective_start, label_end);

    let label_columns = vec![label_metadata.output_column.clone()];
    let label_table = load_output_table(
        &config.label_root,
        request.asset_class,
        request.frequency,
        DEFAULT_BARRA_MODEL,
        false,
        &all_dates,
        &label_columns,
    )?;
    let instruments = instruments_from_table(&label_table)?;
    let universe = load_universe_plan(config, &request.universe, &target_dates, &instruments)?;
    let trade_filter = load_trade_filter_plan(config, request, &target_dates, &instruments)?;
    let benchmark = load_benchmark_plan(config, &request.benchmark, &target_dates, &instruments)?;

    let barra_columns = request.neutralize.barra_columns();
    let barra_table = if !barra_columns.is_empty() {
        Some(load_output_table(
            &config.barra_root,
            request.asset_class,
            request.frequency,
            DEFAULT_BARRA_MODEL,
            true,
            &target_dates,
            &barra_columns,
        )?)
    } else {
        None
    };

    let sector_map = if request.neutralize.uses_industry() {
        let table = read_parquet(
            &config.stock_sw_classification_path,
            Some(&[
                "ts_code".to_string(),
                "in_date".to_string(),
                "out_date".to_string(),
                "l1_code".to_string(),
            ]),
        )?;
        Some(ClassificationMap::from_table(
            &table,
            ClassificationLevel::Sector,
        )?)
    } else {
        None
    };

    Ok(BacktestDataPlan {
        factor_metadata,
        label_metadata,
        target_dates,
        all_dates,
        instruments,
        label_table,
        barra_columns,
        barra_table,
        sector_map,
        universe,
        trade_filter,
        benchmark,
    })
}

pub fn load_backtest_input_batch(
    config: &EngineConfig,
    request: &BacktestRunRequest,
    plan: &BacktestDataPlan,
    factor_metadata: &[FactorMetadata],
    target_dates: &[i32],
    factor_fill_state: &mut FactorFillState,
) -> Result<BacktestInput> {
    if target_dates.is_empty() {
        return Err(err("cannot load empty backtest date batch"));
    }
    let factor_metadata = factor_metadata.to_vec();
    let all_dates = all_dates_for_batch(plan, target_dates)?;
    let factor_columns = factor_metadata
        .iter()
        .map(|row| row.output_column.clone())
        .collect::<Vec<_>>();
    let label_columns = vec![plan.label_metadata.output_column.clone()];
    let factor_root = request
        .factor_root
        .as_deref()
        .unwrap_or(&config.factor_root);
    let factor_layout = factor_root_layout(factor_root, request.asset_class, request.frequency);
    let factor_load = load_factor_output_table_with_presence(
        factor_root,
        factor_layout,
        request.asset_class,
        request.frequency,
        target_dates,
        &factor_columns,
    )?;
    if request.factor_fill.is_forward_fill() {
        initialize_factor_fill_state(
            factor_root,
            factor_layout,
            request.asset_class,
            request.frequency,
            target_dates[0],
            &factor_columns,
            &plan.instruments,
            factor_fill_state,
        )?;
    }

    let label_table = plan.label_table.filter_i32_range(
        "trade_date",
        *all_dates.first().expect("date batch has dates"),
        *all_dates.last().expect("date batch has dates"),
    )?;
    let mut tables = vec![factor_load.table, label_table];
    if let Some(table) = &plan.barra_table {
        tables.push(table.filter_i32_range(
            "trade_date",
            *target_dates.first().expect("target date batch has dates"),
            *target_dates.last().expect("target date batch has dates"),
        )?);
    }
    let mut panel = BacktestPanel::from_tables_with_instruments(
        all_dates.clone(),
        plan.instruments.clone(),
        tables,
    )?;
    panel.ensure_columns(&factor_columns);
    panel.ensure_columns(&label_columns);
    panel.ensure_columns(&plan.barra_columns);
    if request.factor_fill.is_forward_fill() {
        apply_factor_forward_fill(
            &mut panel,
            target_dates,
            &factor_columns,
            &factor_load.present_dates,
            factor_fill_state,
        )?;
    }

    let sectors = if let Some(sector_map) = &plan.sector_map {
        let mut by_date = HashMap::new();
        for date in target_dates {
            by_date.insert(*date, sector_map.groups_for(*date, panel.instruments()));
        }
        Some(by_date)
    } else {
        None
    };

    Ok(BacktestInput {
        factor_metadata,
        label_metadata: plan.label_metadata.clone(),
        target_dates: target_dates.to_vec(),
        all_dates,
        panel,
        universe: plan.universe.slice(target_dates),
        trade_filter: plan.trade_filter.slice(target_dates),
        benchmark: plan.benchmark.slice(target_dates),
        sectors,
    })
}

impl BacktestUniversePlan {
    fn slice(&self, dates: &[i32]) -> BacktestUniverseBatch {
        BacktestUniverseBatch {
            id: self.id.clone(),
            masks: dates
                .iter()
                .filter_map(|date| self.masks.get(date).map(|mask| (*date, mask.clone())))
                .collect(),
        }
    }
}

impl BacktestTradeFilterPlan {
    fn slice(&self, dates: &[i32]) -> BacktestTradeFilterBatch {
        BacktestTradeFilterBatch {
            masks: dates
                .iter()
                .filter_map(|date| self.masks.get(date).map(|mask| (*date, mask.clone())))
                .collect(),
        }
    }
}

impl BenchmarkPlan {
    fn slice(&self, dates: &[i32]) -> BenchmarkBatch {
        let kind = match &self.kind {
            BenchmarkKind::MarketMean => BenchmarkKind::MarketMean,
            BenchmarkKind::Weighted(weights) => BenchmarkKind::Weighted(
                dates
                    .iter()
                    .filter_map(|date| weights.get(date).map(|values| (*date, values.clone())))
                    .collect(),
            ),
        };
        BenchmarkBatch {
            id: self.id.clone(),
            kind,
        }
    }
}

impl BacktestPanel {
    fn from_tables_with_instruments(
        dates: Vec<i32>,
        instruments: Vec<String>,
        tables: Vec<Table>,
    ) -> Result<Self> {
        let date_lookup = dates
            .iter()
            .enumerate()
            .map(|(idx, date)| (*date, idx))
            .collect::<BTreeMap<_, _>>();
        let instrument_lookup = instruments
            .iter()
            .enumerate()
            .map(|(idx, code)| (code.clone(), idx))
            .collect::<BTreeMap<_, _>>();
        let shape_len = dates.len() * instruments.len();
        let mut columns = BTreeMap::<String, Vec<Option<f64>>>::new();
        let mut presence = BTreeMap::<String, Vec<bool>>::new();

        for table in tables {
            if table.columns.is_empty() {
                continue;
            }
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_dates = table.required_i32("trade_date")?;
            let numeric_columns = table
                .columns
                .keys()
                .filter(|name| name.as_str() != "trade_date" && name.as_str() != "ts_code")
                .cloned()
                .collect::<Vec<_>>();
            let numeric_values = numeric_columns
                .iter()
                .map(|name| Ok((name.clone(), table.required_f64_cast(name)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            for name in &numeric_columns {
                columns
                    .entry(name.clone())
                    .or_insert_with(|| vec![None; shape_len]);
                presence
                    .entry(name.clone())
                    .or_insert_with(|| vec![false; shape_len]);
            }
            for row_idx in 0..table.len {
                let (Some(trade_date), Some(ts_code)) =
                    (trade_dates[row_idx], ts_codes[row_idx].clone())
                else {
                    continue;
                };
                let (Some(date_idx), Some(instrument_idx)) = (
                    date_lookup.get(&trade_date),
                    instrument_lookup.get(&ts_code),
                ) else {
                    continue;
                };
                let offset = date_idx * instruments.len() + instrument_idx;
                for name in &numeric_columns {
                    let values = &numeric_values[name];
                    if let Some(target) = columns.get_mut(name) {
                        target[offset] = values.get(row_idx).copied().unwrap_or(None);
                    }
                    if let Some(target) = presence.get_mut(name) {
                        target[offset] = true;
                    }
                }
            }
        }

        Ok(Self {
            dates,
            instruments,
            date_lookup,
            columns,
            presence,
        })
    }
}

fn instruments_from_table(table: &Table) -> Result<Vec<String>> {
    let mut instrument_set = BTreeSet::new();
    let ts_codes = table.required_utf8("ts_code")?;
    for ts_code in ts_codes.iter().flatten() {
        if is_backtest_excluded_instrument(ts_code) {
            continue;
        }
        instrument_set.insert(ts_code.clone());
    }
    Ok(instrument_set.into_iter().collect())
}

fn is_backtest_excluded_instrument(ts_code: &str) -> bool {
    ts_code.to_ascii_uppercase().ends_with(".BJ")
}

#[derive(Clone, Debug)]
struct WeightRecord {
    trade_date: i32,
    ts_code: String,
    weight: Option<f64>,
}

fn load_universe_plan(
    config: &EngineConfig,
    universe_id: &str,
    target_dates: &[i32],
    instruments: &[String],
) -> Result<BacktestUniversePlan> {
    if is_market_all_universe(universe_id) || universe_id.eq_ignore_ascii_case("000985.CSI") {
        return Ok(BacktestUniversePlan {
            id: universe_id.to_string(),
            masks: target_dates
                .iter()
                .map(|date| (*date, vec![true; instruments.len()]))
                .collect(),
        });
    }
    let records = if is_builtin_index(universe_id) {
        load_index_weight_records(config, universe_id, target_dates)?
    } else {
        load_custom_universe_records(config, universe_id, false)?
    };
    let weights = effective_weights_by_date(&records, target_dates, instruments);
    let masks = weights
        .into_iter()
        .map(|(date, values)| {
            (
                date,
                values
                    .into_iter()
                    .map(|weight| weight.is_some_and(|value| value.is_finite() && value > 0.0))
                    .collect(),
            )
        })
        .collect();
    Ok(BacktestUniversePlan {
        id: universe_id.to_string(),
        masks,
    })
}

fn load_trade_filter_masks(
    config: &EngineConfig,
    request: &BacktestRunRequest,
    target_dates: &[i32],
    instruments: &[String],
) -> Result<HashMap<i32, Vec<bool>>> {
    let instrument_lookup = instruments
        .iter()
        .enumerate()
        .map(|(idx, code)| (code.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let root = config
        .data_root
        .join("stock_data")
        .join("daily")
        .join("trade_filter");
    let mut masks = HashMap::new();
    for date in target_dates {
        let required = request.exclude_limit || (request.exclude_st && *date >= 20160101);
        let path = daily_trade_filter_path(&root, *date);
        if !path.exists() {
            if required {
                return Err(err(format!(
                    "missing stock trade filter for {date}: expected {}. Run: python scripts\\update_incremental.py --groups stock_trade_filter --start-date {date} --end-date {date}",
                    path.display()
                )));
            }
            masks.insert(*date, vec![true; instruments.len()]);
            continue;
        }
        let table = read_parquet(
            &path,
            Some(&[
                "trade_date".to_string(),
                "ts_code".to_string(),
                "is_limit_up".to_string(),
                "is_limit_down".to_string(),
                "is_limit".to_string(),
                "is_st".to_string(),
            ]),
        )?;
        let trade_dates = table.required_i32_date_cast("trade_date")?;
        let ts_codes = table.required_utf8("ts_code")?;
        let is_limit_up = bool_column_cast(&table, "is_limit_up")?;
        let is_limit_down = bool_column_cast(&table, "is_limit_down")?;
        let is_limit = bool_column_cast(&table, "is_limit")?;
        let is_st = bool_column_cast(&table, "is_st")?;
        let mut mask = vec![true; instruments.len()];
        for row_idx in 0..table.len {
            if trade_dates[row_idx] != Some(*date) {
                continue;
            }
            let Some(ts_code) = ts_codes[row_idx].as_deref() else {
                continue;
            };
            let Some(instrument_idx) = instrument_lookup.get(ts_code).copied() else {
                continue;
            };
            let limit_allowed = !request.exclude_limit
                || request.limit_side.allows(
                    is_limit_up[row_idx].unwrap_or(false),
                    is_limit_down[row_idx].unwrap_or(false),
                    is_limit[row_idx].unwrap_or(false),
                );
            let st_allowed =
                !request.exclude_st || *date < 20160101 || !is_st[row_idx].unwrap_or(false);
            mask[instrument_idx] = limit_allowed && st_allowed;
        }
        masks.insert(*date, mask);
    }
    Ok(masks)
}

fn load_trade_filter_plan(
    config: &EngineConfig,
    request: &BacktestRunRequest,
    target_dates: &[i32],
    instruments: &[String],
) -> Result<BacktestTradeFilterPlan> {
    let masks = if request.exclude_limit || request.exclude_st {
        load_trade_filter_masks(config, request, target_dates, instruments)?
    } else {
        target_dates
            .iter()
            .map(|date| (*date, vec![true; instruments.len()]))
            .collect()
    };
    Ok(BacktestTradeFilterPlan { masks })
}

fn daily_trade_filter_path(root: &Path, trade_date: i32) -> PathBuf {
    let year = trade_date / 10_000;
    root.join(year.to_string())
        .join(format!("{trade_date}.parquet"))
}

fn bool_column_cast(table: &Table, name: &str) -> Result<Vec<Option<bool>>> {
    match table.columns.get(name) {
        Some(ColumnData::Bool(values)) => Ok(values.clone()),
        Some(ColumnData::I32(values)) => {
            Ok(values.iter().map(|value| value.map(|v| v != 0)).collect())
        }
        Some(ColumnData::I64(values)) => {
            Ok(values.iter().map(|value| value.map(|v| v != 0)).collect())
        }
        Some(ColumnData::F32(values)) => Ok(values
            .iter()
            .map(|value| value.map(|v| v.is_finite() && v != 0.0))
            .collect()),
        Some(ColumnData::F64(values)) => Ok(values
            .iter()
            .map(|value| value.map(|v| v.is_finite() && v != 0.0))
            .collect()),
        Some(_) => Err(err(format!("column {name} cannot be cast to bool"))),
        None => Err(err(format!("missing required column {name}"))),
    }
}

fn load_benchmark_plan(
    config: &EngineConfig,
    benchmark_id: &str,
    target_dates: &[i32],
    instruments: &[String],
) -> Result<BenchmarkPlan> {
    if benchmark_id.eq_ignore_ascii_case("mkt_mean") {
        return Ok(BenchmarkPlan {
            id: benchmark_id.to_string(),
            kind: BenchmarkKind::MarketMean,
        });
    }
    let records = if is_builtin_index(benchmark_id) {
        load_index_weight_records(config, benchmark_id, target_dates)?
    } else {
        load_custom_universe_records(config, benchmark_id, true)?
    };
    Ok(BenchmarkPlan {
        id: benchmark_id.to_string(),
        kind: BenchmarkKind::Weighted(effective_weights_by_date(
            &records,
            target_dates,
            instruments,
        )),
    })
}

fn is_market_all_universe(value: &str) -> bool {
    value.eq_ignore_ascii_case("mkt_all") || value.eq_ignore_ascii_case("all")
}

fn is_builtin_index(value: &str) -> bool {
    matches!(
        value.to_ascii_uppercase().as_str(),
        "000300.SH" | "000905.SH" | "000852.SH" | "000985.CSI"
    )
}

fn universe_list_date_floor(value: &str) -> Option<i32> {
    match value.to_ascii_uppercase().as_str() {
        "000300.SH" => Some(20050408),
        "000905.SH" => Some(20070115),
        "000985.CSI" => Some(20110802),
        "000852.SH" => Some(20141017),
        _ => None,
    }
}

fn load_index_weight_records(
    config: &EngineConfig,
    index_code: &str,
    target_dates: &[i32],
) -> Result<Vec<WeightRecord>> {
    let first = target_dates
        .first()
        .copied()
        .ok_or_else(|| err("cannot load index weights for empty date range"))?;
    let last = target_dates
        .last()
        .copied()
        .ok_or_else(|| err("cannot load index weights for empty date range"))?;
    let code_dir = config
        .data_root
        .join("index_data")
        .join("monthly_weight")
        .join(index_code.replace('.', "_"));
    if !code_dir.exists() {
        return Err(err(format!(
            "missing index weight data for {index_code}: expected {}. Run: python scripts\\update_incremental.py --groups index_weight --ts-code {index_code}",
            code_dir.display()
        )));
    }
    let mut records = Vec::new();
    for path in collect_parquet_files_recursive(&code_dir)? {
        if !index_weight_path_may_overlap(&path, first, last) {
            continue;
        };
        let table = read_parquet(
            &path,
            Some(&[
                "trade_date".to_string(),
                "con_code".to_string(),
                "weight".to_string(),
            ]),
        )?;
        let dates = table.required_i32_date_cast("trade_date")?;
        let codes = table.required_utf8("con_code")?;
        let weights = table.required_f64_cast("weight")?;
        for row_idx in 0..table.len {
            let (Some(trade_date), Some(ts_code)) = (dates[row_idx], codes[row_idx].clone()) else {
                continue;
            };
            records.push(WeightRecord {
                trade_date,
                ts_code,
                weight: weights[row_idx].map(index_weight_to_decimal),
            });
        }
    }
    if records.is_empty() {
        return Err(err(format!(
            "index weight directory for {index_code} contains no usable rows: {}",
            code_dir.display()
        )));
    }
    Ok(records)
}

fn index_weight_to_decimal(weight_percent: f64) -> f64 {
    weight_percent / 100.0
}

fn collect_parquet_files_recursive(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_parquet_files_recursive_into(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_parquet_files_recursive_into(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet_files_recursive_into(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
            files.push(path);
        }
    }
    Ok(())
}

fn index_weight_path_may_overlap(path: &Path, first: i32, last: i32) -> bool {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return true;
    };
    if stem.len() == 4 {
        if let Ok(year) = stem.parse::<i32>() {
            return year >= first / 10_000 - 1 && year <= last / 10_000;
        }
    }
    if stem.len() == 6 {
        if let Ok(year_month) = stem.parse::<i32>() {
            let first_month = first / 100;
            let last_month = last / 100;
            return year_month >= first_month - 1 && year_month <= last_month;
        }
    }
    true
}

fn load_custom_universe_records(
    config: &EngineConfig,
    universe_id: &str,
    require_weight: bool,
) -> Result<Vec<WeightRecord>> {
    let path = config
        .data_root
        .join("universe")
        .join(format!("{universe_id}.parquet"));
    if !path.exists() {
        return Err(err(format!(
            "missing custom universe {universe_id}: expected {}",
            path.display()
        )));
    }
    let table = read_parquet(&path, None)?;
    let dates = table.required_i32_date_cast("trade_date")?;
    let codes = table.required_utf8("ts_code")?;
    let weights = if table.columns.contains_key("weight") {
        Some(table.required_f64_cast("weight")?)
    } else if require_weight {
        return Err(err(format!(
            "custom benchmark {universe_id} must include a numeric weight column: {}",
            path.display()
        )));
    } else {
        None
    };
    let mut records = Vec::new();
    for row_idx in 0..table.len {
        let (Some(trade_date), Some(ts_code)) = (dates[row_idx], codes[row_idx].clone()) else {
            continue;
        };
        records.push(WeightRecord {
            trade_date,
            ts_code,
            weight: weights
                .as_ref()
                .map(|values| values[row_idx])
                .unwrap_or(Some(1.0)),
        });
    }
    if records.is_empty() {
        return Err(err(format!(
            "custom universe {universe_id} contains no usable rows: {}",
            path.display()
        )));
    }
    Ok(records)
}

fn effective_weights_by_date(
    records: &[WeightRecord],
    target_dates: &[i32],
    instruments: &[String],
) -> HashMap<i32, Vec<Option<f64>>> {
    let instrument_lookup = instruments
        .iter()
        .enumerate()
        .map(|(idx, code)| (code.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let mut records_by_date = BTreeMap::<i32, BTreeMap<String, f64>>::new();
    for record in records {
        let Some(weight) = record
            .weight
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        records_by_date
            .entry(record.trade_date)
            .or_default()
            .insert(record.ts_code.clone(), weight);
    }

    let mut output = HashMap::new();
    let mut current = BTreeMap::<String, f64>::new();
    let mut iter = records_by_date.into_iter().peekable();
    for date in target_dates {
        while iter
            .peek()
            .is_some_and(|(effective_date, _)| effective_date <= date)
        {
            let (_, weights) = iter.next().expect("peeked record");
            current = weights;
        }
        let mut values = vec![None; instruments.len()];
        for (code, weight) in &current {
            if let Some(idx) = instrument_lookup.get(code.as_str()) {
                values[*idx] = Some(*weight);
            }
        }
        output.insert(*date, values);
    }
    output
}

fn all_dates_for_batch(plan: &BacktestDataPlan, target_dates: &[i32]) -> Result<Vec<i32>> {
    let first = *target_dates
        .first()
        .ok_or_else(|| err("cannot build all_dates for empty target date batch"))?;
    let last = *target_dates
        .last()
        .ok_or_else(|| err("cannot build all_dates for empty target date batch"))?;
    let start_idx = plan
        .all_dates
        .iter()
        .position(|date| *date == first)
        .ok_or_else(|| err(format!("date {first} not found in backtest calendar")))?;
    let end_idx = plan
        .all_dates
        .iter()
        .position(|date| *date == last)
        .ok_or_else(|| err(format!("date {last} not found in backtest calendar")))?;
    let forward_days = plan.label_metadata.lookahead.max(1);
    let label_end_idx = (end_idx + forward_days).min(plan.all_dates.len() - 1);
    Ok(plan.all_dates[start_idx..=label_end_idx].to_vec())
}

fn select_factors(
    config: &EngineConfig,
    request: &BacktestRunRequest,
) -> Result<Vec<FactorMetadata>> {
    if let Some(root) = &request.factor_root {
        if !root.exists() {
            return Err(err(format!(
                "external factor root not found: {}",
                root.display()
            )));
        }
        if request.tags.is_some() {
            return Err(err(
                "--tags cannot be used with --factor-root; use --factors or --all-factors",
            ));
        }
        let ids = match (&request.factor_ids, request.all_factors) {
            (Some(ids), false) => ids.clone(),
            (None, true) => {
                external_factor_ids_from_root(root, request.asset_class, request.frequency)?
            }
            (None, false) => {
                return Err(err(
                    "--factor-root requires --factors factor_id[,factor_id...] or --all-factors",
                ));
            }
            _ => {
                return Err(err(
                    "--factors, --tags and --all-factors cannot be used together",
                ));
            }
        };
        let rows = ids
            .iter()
            .map(|id| external_factor_metadata(id, request.asset_class, request.frequency))
            .collect::<Vec<_>>();
        return Ok(dedup_factor_metadata(rows));
    }

    let storage = FactorStorage::new(config.factor_root.clone());
    let metadata = storage.read_metadata()?;
    let selected = match (&request.factor_ids, &request.tags, request.all_factors) {
        (Some(ids), None, false) => {
            let mut rows = Vec::new();
            for id in ids {
                let Some(row) = metadata.iter().find(|row| &row.factor_id == id) else {
                    return Err(err(format!("factor not found in metadata: {id}")));
                };
                if row.tags.iter().any(|tag| tag == "deprecated") {
                    return Err(err(format!(
                        "deprecated factor cannot be backtested explicitly: {id}"
                    )));
                }
                rows.push(row.clone());
            }
            rows
        }
        (None, Some(tags), false) => metadata
            .into_iter()
            .filter(|row| !row.tags.iter().any(|tag| tag == "deprecated"))
            .filter(|row| {
                tags.iter()
                    .all(|tag| row.tags.iter().any(|row_tag| row_tag == tag))
            })
            .collect(),
        (None, None, true) => metadata
            .into_iter()
            .filter(|row| !row.tags.iter().any(|tag| tag == "deprecated"))
            .collect(),
        (None, None, false) => {
            return Err(err("backtest requires --factors, --tags or --all-factors"));
        }
        _ => {
            return Err(err(
                "--factors, --tags and --all-factors cannot be used together",
            ));
        }
    };
    Ok(dedup_factor_metadata(selected))
}

const EXTERNAL_FACTOR_KEY_COLUMNS: &[&str] = &["trade_date", "trade_time", "ts_code"];

fn external_factor_ids_from_root(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
) -> Result<Vec<String>> {
    let layout = factor_root_layout(root, asset_class, frequency);
    let base = factor_root_base_path(root, layout, asset_class, frequency);
    let mut columns = BTreeSet::new();
    collect_external_factor_columns(&base, &mut columns)?;
    columns.retain(|column| !EXTERNAL_FACTOR_KEY_COLUMNS.contains(&column.as_str()));
    if columns.is_empty() {
        return Err(err(format!(
            "no factor columns found in external factor root: {}",
            base.display()
        )));
    }
    Ok(columns.into_iter().collect())
}

fn collect_external_factor_columns(path: &Path, columns: &mut BTreeSet<String>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_external_factor_columns(&path, columns)?;
            continue;
        }
        let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
            continue;
        };
        if !extension.eq_ignore_ascii_case("parquet") {
            continue;
        }
        columns.extend(parquet_column_names(&path)?);
    }
    Ok(())
}

fn external_factor_metadata(
    factor_id: &str,
    asset_class: AssetClass,
    frequency: Frequency,
) -> FactorMetadata {
    FactorMetadata {
        factor_id: factor_id.to_string(),
        aliases: Vec::new(),
        aliases_json: "[]".to_string(),
        version: "external".to_string(),
        output_column: factor_id.to_string(),
        name: factor_id.to_string(),
        asset_class: asset_class.as_str().to_string(),
        frequency: frequency.as_str().to_string(),
        tags: vec!["external".to_string()],
        tags_json: r#"["external"]"#.to_string(),
        dependencies_json: "[]".to_string(),
        description: "External factor supplied via --factor-root.".to_string(),
        updated_at: String::new(),
    }
}

fn dedup_factor_metadata(rows: Vec<FactorMetadata>) -> Vec<FactorMetadata> {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for row in rows {
        if seen.insert(row.factor_id.clone()) {
            output.push(row);
        }
    }
    output
}

fn select_label(config: &EngineConfig, label_id: &str) -> Result<LabelMetadataInfo> {
    let storage = LabelStorage::new(config.label_root.clone());
    let metadata = storage.read_metadata()?;
    let row = metadata
        .iter()
        .find(|row| row.label_id == label_id)
        .ok_or_else(|| err(format!("label not found in metadata: {label_id}")))?;
    Ok(LabelMetadataInfo {
        label_id: row.label_id.clone(),
        output_column: row.output_column.clone(),
        lookahead: parse_lookahead(&row.dependencies_json).unwrap_or(2),
    })
}

fn parse_lookahead(dependencies_json: &str) -> Option<usize> {
    let marker = "\"lookahead_trading_days\":";
    let start = dependencies_json.find(marker)? + marker.len();
    let tail = &dependencies_json[start..];
    let digits = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse::<usize>().ok()
}

fn load_output_table(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
    model: &str,
    barra_layout: bool,
    dates: &[i32],
    requested_columns: &[String],
) -> Result<Table> {
    let mut columns = vec!["trade_date".to_string(), "ts_code".to_string()];
    for column in requested_columns {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.clone());
        }
    }
    let mut table = Table::empty();
    for date in dates {
        let path = output_path(root, asset_class, frequency, model, barra_layout, *date);
        if !path.exists() {
            continue;
        }
        let mut daily = read_parquet(&path, Some(&columns))?;
        ensure_table_columns(&mut daily, requested_columns)?;
        if table.columns.is_empty() {
            table = daily;
        } else {
            table.append(&daily)?;
        }
    }
    if table.columns.is_empty() {
        empty_output_table(&columns)
    } else {
        Ok(table)
    }
}

fn load_factor_output_table_with_presence(
    root: &Path,
    layout: FactorRootLayout,
    asset_class: AssetClass,
    frequency: Frequency,
    dates: &[i32],
    requested_columns: &[String],
) -> Result<FactorLoadResult> {
    let mut columns = vec!["trade_date".to_string(), "ts_code".to_string()];
    for column in requested_columns {
        if !columns.iter().any(|existing| existing == column) {
            columns.push(column.clone());
        }
    }
    let mut table = Table::empty();
    let mut present_dates = requested_columns
        .iter()
        .map(|column| (column.clone(), BTreeSet::<i32>::new()))
        .collect::<BTreeMap<_, _>>();
    for date in dates {
        let path = factor_output_path(root, layout, asset_class, frequency, *date);
        if !path.exists() {
            continue;
        }
        let schema_columns = parquet_column_names(&path)?;
        for column in requested_columns {
            if schema_columns.contains(column) {
                if let Some(dates) = present_dates.get_mut(column) {
                    dates.insert(*date);
                }
            }
        }
        let mut daily = read_parquet(&path, Some(&columns))?;
        ensure_table_columns(&mut daily, requested_columns)?;
        if table.columns.is_empty() {
            table = daily;
        } else {
            table.append(&daily)?;
        }
    }
    let table = if table.columns.is_empty() {
        empty_output_table(&columns)?
    } else {
        table
    };
    Ok(FactorLoadResult {
        table,
        present_dates,
    })
}

fn initialize_factor_fill_state(
    root: &Path,
    layout: FactorRootLayout,
    asset_class: AssetClass,
    frequency: Frequency,
    first_target_date: i32,
    factor_columns: &[String],
    instruments: &[String],
    state: &mut FactorFillState,
) -> Result<()> {
    if state.initialized {
        return Ok(());
    }
    state.initialized = true;
    let mut remaining = factor_columns.iter().cloned().collect::<BTreeSet<_>>();
    if remaining.is_empty() {
        return Ok(());
    }
    let mut dates =
        available_factor_dates_before(root, layout, asset_class, frequency, first_target_date)?;
    dates.sort_by(|left, right| right.cmp(left));
    for date in dates {
        if remaining.is_empty() {
            break;
        }
        let path = factor_output_path(root, layout, asset_class, frequency, date);
        if !path.exists() {
            continue;
        }
        let schema_columns = parquet_column_names(&path)?;
        let columns_to_load = remaining
            .iter()
            .filter(|column| schema_columns.contains(*column))
            .cloned()
            .collect::<Vec<_>>();
        if columns_to_load.is_empty() {
            continue;
        }
        update_factor_fill_state_from_path(&path, date, &columns_to_load, instruments, state)?;
        for column in columns_to_load {
            remaining.remove(&column);
        }
    }
    Ok(())
}

fn available_factor_dates_before(
    root: &Path,
    layout: FactorRootLayout,
    asset_class: AssetClass,
    frequency: Frequency,
    first_target_date: i32,
) -> Result<Vec<i32>> {
    let base = factor_root_base_path(root, layout, asset_class, frequency);
    let mut dates = Vec::new();
    collect_parquet_dates_before(&base, first_target_date, &mut dates)?;
    dates.sort_unstable();
    dates.dedup();
    Ok(dates)
}

fn factor_root_base_path(
    root: &Path,
    layout: FactorRootLayout,
    asset_class: AssetClass,
    frequency: Frequency,
) -> PathBuf {
    match layout {
        FactorRootLayout::Standard => root.join(asset_class.as_str()).join(frequency.as_str()),
        FactorRootLayout::DirectDaily => root.to_path_buf(),
    }
}

fn collect_parquet_dates_before(
    path: &Path,
    first_target_date: i32,
    dates: &mut Vec<i32>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_parquet_dates_before(&path, first_target_date, dates)?;
            continue;
        }
        let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
            continue;
        };
        if !extension.eq_ignore_ascii_case("parquet") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        let Ok(date) = stem.parse::<i32>() else {
            continue;
        };
        if date < first_target_date {
            dates.push(date);
        }
    }
    Ok(())
}

fn update_factor_fill_state_from_path(
    path: &Path,
    trade_date: i32,
    factor_columns: &[String],
    instruments: &[String],
    state: &mut FactorFillState,
) -> Result<()> {
    let mut columns = vec!["trade_date".to_string(), "ts_code".to_string()];
    for column in factor_columns {
        columns.push(column.clone());
    }
    let table = read_parquet(path, Some(&columns))?;
    let trade_dates = table.required_i32_date_cast("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let instrument_lookup = instruments
        .iter()
        .enumerate()
        .map(|(idx, code)| (code.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let values_by_column = factor_columns
        .iter()
        .map(|column| Ok((column.clone(), table.required_f64_cast(column)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;

    for column in factor_columns {
        state
            .latest
            .entry(column.clone())
            .or_insert_with(|| vec![None; instruments.len()]);
    }
    for row_idx in 0..table.len {
        if trade_dates[row_idx] != Some(trade_date) {
            continue;
        }
        let Some(ts_code) = ts_codes[row_idx].as_deref() else {
            continue;
        };
        let Some(instrument_idx) = instrument_lookup.get(ts_code).copied() else {
            continue;
        };
        for column in factor_columns {
            let value = values_by_column[column]
                .get(row_idx)
                .copied()
                .unwrap_or(None);
            if let Some(latest) = state.latest.get_mut(column) {
                latest[instrument_idx] = value;
            }
        }
    }
    Ok(())
}

fn apply_factor_forward_fill(
    panel: &mut BacktestPanel,
    target_dates: &[i32],
    factor_columns: &[String],
    present_dates: &BTreeMap<String, BTreeSet<i32>>,
    state: &mut FactorFillState,
) -> Result<()> {
    let instrument_count = panel.instruments.len();
    for date in target_dates {
        let Some(date_idx) = panel.date_index(*date) else {
            continue;
        };
        let start = date_idx * instrument_count;
        let end = start + instrument_count;
        for column in factor_columns {
            state
                .latest
                .entry(column.clone())
                .or_insert_with(|| vec![None; instrument_count]);
            let has_real_snapshot = present_dates
                .get(column)
                .is_some_and(|dates| dates.contains(date));
            let values = panel
                .columns
                .get_mut(column)
                .ok_or_else(|| err(format!("backtest panel missing factor column {column}")))?;
            let presence = panel.presence.get_mut(column).ok_or_else(|| {
                err(format!(
                    "backtest panel missing factor presence for {column}"
                ))
            })?;
            let latest = state
                .latest
                .get_mut(column)
                .expect("factor fill state was initialized");
            if has_real_snapshot {
                latest.copy_from_slice(&values[start..end]);
            } else {
                for idx in 0..instrument_count {
                    let value = latest[idx];
                    values[start + idx] = value;
                    presence[start + idx] = value.is_some();
                }
            }
        }
    }
    Ok(())
}

fn ensure_table_columns(table: &mut Table, requested_columns: &[String]) -> Result<()> {
    for column in requested_columns {
        if !table.columns.contains_key(column) {
            table.insert(column.clone(), ColumnData::F64(vec![None; table.len]))?;
        }
    }
    Ok(())
}

fn factor_root_layout(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
) -> FactorRootLayout {
    if root
        .join(asset_class.as_str())
        .join(frequency.as_str())
        .exists()
    {
        FactorRootLayout::Standard
    } else {
        FactorRootLayout::DirectDaily
    }
}

fn factor_output_path(
    root: &Path,
    layout: FactorRootLayout,
    asset_class: AssetClass,
    frequency: Frequency,
    trade_date: i32,
) -> PathBuf {
    let year = trade_date / 10_000;
    match layout {
        FactorRootLayout::Standard => root
            .join(asset_class.as_str())
            .join(frequency.as_str())
            .join(year.to_string())
            .join(format!("{trade_date}.parquet")),
        FactorRootLayout::DirectDaily => root
            .join(year.to_string())
            .join(format!("{trade_date}.parquet")),
    }
}

fn output_path(
    root: &Path,
    asset_class: AssetClass,
    frequency: Frequency,
    model: &str,
    barra_layout: bool,
    trade_date: i32,
) -> PathBuf {
    let year = trade_date / 10_000;
    if barra_layout {
        root.join(asset_class.as_str())
            .join(frequency.as_str())
            .join(model)
            .join(year.to_string())
            .join(format!("{trade_date}.parquet"))
    } else {
        root.join(asset_class.as_str())
            .join(frequency.as_str())
            .join(year.to_string())
            .join(format!("{trade_date}.parquet"))
    }
}

fn empty_output_table(columns: &[String]) -> Result<Table> {
    let mut data = BTreeMap::new();
    for column in columns {
        let values = if column == "ts_code" {
            ColumnData::Utf8(Vec::new())
        } else if column == "trade_date" {
            ColumnData::I32(Vec::new())
        } else {
            ColumnData::F64(Vec::new())
        };
        data.insert(column.clone(), values);
    }
    Table::new(data)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_factor_forward_fill, effective_weights_by_date, external_factor_ids_from_root,
        index_weight_path_may_overlap, index_weight_to_decimal, instruments_from_table,
        parse_lookahead, universe_list_date_floor, BacktestPanel, FactorFillState, WeightRecord,
    };
    use crate::core::{AssetClass, Frequency};
    use crate::data::parquet_io::write_parquet;
    use crate::data::{ColumnData, Table};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    #[test]
    fn parses_label_lookahead_from_metadata_json() {
        assert_eq!(
            parse_lookahead(r#"[{"dataset":"x"},{"lookahead_trading_days":2}]"#),
            Some(2)
        );
    }

    #[test]
    fn builtin_universe_list_dates_are_known() {
        assert_eq!(universe_list_date_floor("000300.SH"), Some(20050408));
        assert_eq!(universe_list_date_floor("000985.CSI"), Some(20110802));
        assert_eq!(universe_list_date_floor("mkt_all"), None);
    }

    #[test]
    fn effective_weights_forward_fill_latest_membership() {
        let records = vec![
            WeightRecord {
                trade_date: 20240101,
                ts_code: "000001.SZ".to_string(),
                weight: Some(60.0),
            },
            WeightRecord {
                trade_date: 20240101,
                ts_code: "000002.SZ".to_string(),
                weight: Some(40.0),
            },
            WeightRecord {
                trade_date: 20240201,
                ts_code: "000002.SZ".to_string(),
                weight: Some(100.0),
            },
        ];
        let instruments = vec!["000001.SZ".to_string(), "000002.SZ".to_string()];
        let weights = effective_weights_by_date(&records, &[20240115, 20240215], &instruments);

        assert_eq!(weights[&20240115], vec![Some(60.0), Some(40.0)]);
        assert_eq!(weights[&20240215], vec![None, Some(100.0)]);
    }

    #[test]
    fn index_weight_percent_is_converted_to_decimal() {
        assert_eq!(index_weight_to_decimal(60.0), 0.6);
        assert_eq!(index_weight_to_decimal(2.5), 0.025);
    }

    #[test]
    fn index_weight_path_filter_supports_monthly_and_legacy_year_files() {
        assert!(index_weight_path_may_overlap(
            Path::new("000300_SH/2026/202603.parquet"),
            20260101,
            20260331,
        ));
        assert!(index_weight_path_may_overlap(
            Path::new("000300_SH/2026.parquet"),
            20260101,
            20260331,
        ));
        assert!(!index_weight_path_may_overlap(
            Path::new("000300_SH/2024/202401.parquet"),
            20260101,
            20260331,
        ));
    }

    #[test]
    fn external_factor_all_factors_scans_non_key_columns() {
        let root = test_output_dir("external_factor_all_factors_scans_non_key_columns");
        let path = root.join("2026").join("20260424.parquet");
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260424)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            ("bar".to_string(), ColumnData::F64(vec![Some(2.0)])),
            ("foo".to_string(), ColumnData::F64(vec![Some(1.0)])),
        ]))
        .expect("table");
        write_parquet(&path, &table).expect("write external factor parquet");

        let ids = external_factor_ids_from_root(&root, AssetClass::Stock, Frequency::Daily)
            .expect("scan external factors");
        assert_eq!(ids, vec!["bar".to_string(), "foo".to_string()]);
    }

    fn test_output_dir(name: &str) -> PathBuf {
        let path = std::env::current_dir()
            .expect("cwd")
            .join("target")
            .join("backtest_tests")
            .join(name);
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("clean test output dir");
        }
        std::fs::create_dir_all(&path).expect("create test output dir");
        path
    }

    #[test]
    fn backtest_instruments_exclude_bj_codes() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("920087.BJ".to_string()),
                    Some("600000.SH".to_string()),
                ]),
            ),
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260424), Some(20260424), Some(20260424)]),
            ),
        ]))
        .expect("valid table");

        assert_eq!(
            instruments_from_table(&table).expect("instruments"),
            vec!["000001.SZ".to_string(), "600000.SH".to_string()]
        );
    }

    #[test]
    fn factor_forward_fill_fills_missing_date_snapshots() {
        let mut panel = test_panel(vec![Some(1.0), None, None, None]);
        let mut present_dates = BTreeMap::new();
        present_dates.insert("alpha".to_string(), BTreeSet::from([20240101]));
        let mut state = FactorFillState::new(&["alpha".to_string()], 2);

        apply_factor_forward_fill(
            &mut panel,
            &[20240101, 20240102],
            &["alpha".to_string()],
            &present_dates,
            &mut state,
        )
        .expect("ffill");

        let values = panel.columns.get("alpha").expect("alpha");
        assert_eq!(values, &vec![Some(1.0), None, Some(1.0), None]);
    }

    #[test]
    fn factor_forward_fill_does_not_patch_stock_level_nulls_on_real_snapshot() {
        let mut panel = test_panel(vec![None, None, None, Some(2.0)]);
        let mut present_dates = BTreeMap::new();
        present_dates.insert("alpha".to_string(), BTreeSet::from([20240102]));
        let mut state = FactorFillState::new(&["alpha".to_string()], 2);
        state
            .latest
            .insert("alpha".to_string(), vec![Some(9.0), Some(8.0)]);

        apply_factor_forward_fill(
            &mut panel,
            &[20240102],
            &["alpha".to_string()],
            &present_dates,
            &mut state,
        )
        .expect("ffill");

        let values = panel.columns.get("alpha").expect("alpha");
        assert_eq!(values, &vec![None, None, None, Some(2.0)]);
        assert_eq!(state.latest["alpha"], vec![None, Some(2.0)]);
    }

    fn test_panel(alpha_values: Vec<Option<f64>>) -> BacktestPanel {
        let dates = vec![20240101, 20240102];
        let instruments = vec!["000001.SZ".to_string(), "000002.SZ".to_string()];
        let date_lookup = dates
            .iter()
            .enumerate()
            .map(|(idx, date)| (*date, idx))
            .collect::<BTreeMap<_, _>>();
        let presence_values = alpha_values.iter().map(Option::is_some).collect::<Vec<_>>();
        BacktestPanel {
            dates,
            instruments,
            date_lookup,
            columns: BTreeMap::from([("alpha".to_string(), alpha_values)]),
            presence: BTreeMap::from([("alpha".to_string(), presence_values)]),
        }
    }
}

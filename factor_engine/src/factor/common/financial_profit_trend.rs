use std::any::Any;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, ClassificationLevel, ClassificationMap,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const WINDOW: usize = 8;
const EPS: f64 = 1e-12;
const PROVIDER_KEY: &str = "stock|daily|financial_profit_trend";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    Opmd,
    Lpnp,
    Ocfa,
    Qpt,
}
const ALL: [Output; 4] = [Output::Opmd, Output::Lpnp, Output::Ocfa, Output::Qpt];

impl Output {
    pub fn id(self) -> &'static str {
        match self {
            Self::Opmd => "opmd",
            Self::Lpnp => "lpnp",
            Self::Ocfa => "ocfa",
            Self::Qpt => "qpt",
        }
    }

    fn fields(
        self,
    ) -> (
        &'static [&'static str],
        &'static [&'static str],
        &'static [&'static str],
    ) {
        match self {
            Self::Opmd => (&["operate_profit", "revenue"], &[], &[]),
            Self::Lpnp => (
                &["n_income", "non_oper_income"],
                &[],
                &["c_paid_to_for_empl"],
            ),
            Self::Ocfa => (&["total_cogs"], &["fix_assets"], &[]),
            Self::Qpt => (&["n_income_attr_p"], &[], &[]),
        }
    }

    fn quarters(self) -> usize {
        match self {
            Self::Opmd => 5,
            Self::Lpnp | Self::Ocfa => WINDOW,
            // An extra quarter permits the previous eight-quarter acceleration fallback.
            Self::Qpt => WINDOW + 1,
        }
    }
}

fn spec(output: Output) -> FactorSpec {
    let (income, balance, cash) = output.fields();
    let mut dependencies = vec![DataRequest::financial_quarters(
        DatasetId::StockIncome,
        income,
        output.quarters(),
    )];
    for (dataset, fields) in [
        (DatasetId::StockBalanceSheet, balance),
        (DatasetId::StockCashFlow, cash),
    ] {
        if !fields.is_empty() {
            dependencies.push(DataRequest::financial_quarters(dataset, fields, WINDOW));
        }
    }
    dependencies.push(DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]));
    dependencies.push(DataRequest::new(
        DatasetId::StockSwClassification,
        &["l1_code"],
    ));
    let (broker, kind, description) = match output {
        Output::Opmd => ("CICC", "profitability", "Current quarterly-anchor TTM operating profit/revenue minus the previous quarter's TTM margin. Both four-quarter sums must be complete; revenue denominators must be positive."),
        Output::Lpnp => ("GDZQ", "ts_regression", "Latest in-sample residual of single-quarter n_income on non_oper_income and c_paid_to_for_empl. Eight complete consecutive quarters, each variable time-series population-zscored, OLS with intercept; missing data, constant variables or rank deficiency yield null."),
        Output::Ocfa => ("GDZQ", "ts_regression", "Latest in-sample residual of single-quarter total_cogs on same-quarter end fix_assets. Eight complete consecutive quarters, each variable time-series population-zscored, OLS with intercept; missing data or constant variables yield null."),
        Output::Qpt => ("CICC", "growth", "Daily latest disclosed quarter, no month-end freeze. Growth=current parent-profit TTM / previous-quarter parent-profit TTM - 1 with signed nonzero denominator. Acceleration is the raw quadratic coefficient over eight single quarters. Growth tercile score plus within-group acceleration population zscore. Missing growth is forced to group 2, with previous-quarter acceleration fallback if current acceleration is unavailable. All eight acceleration observations required; singleton/constant groups contribute zero acceleration score."),
    };
    let mut tags: Vec<String> = [
        broker,
        "fundamental",
        "financial",
        "pit",
        kind,
        "neutralize",
        "size",
        "sector",
        "daily",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    if output == Output::Opmd {
        tags.push("deprecated".into());
    }
    FactorSpec {
        id: output.id().into(),
        aliases: vec![output.id().to_ascii_uppercase()],
        name: output.id().to_ascii_uppercase(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: "0.1.0".into(),
        tags,
        description: format!("{description} Regular PIT reports only. SW L1 and Barra SIZE neutralized daily; excludes BJ; no winsorization or final zscore."),
        dependencies,
        intraday_raw_dependencies: vec![],
        lookback: Lookback { trading_days: 0 },
    }
}

pub struct FinancialProfitTrend {
    output: Output,
}
impl FinancialProfitTrend {
    pub fn new(output: Output) -> Self {
        Self { output }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Snapshot {
    raw: [Option<f64>; 3],
    growth: Option<f64>,
    acceleration: Option<f64>,
}
#[derive(Default)]
struct ComputeState {
    requested: Vec<Output>,
    snapshots: InstrumentAlignedSnapshotCache<Snapshot>,
}

impl Factor for FinancialProfitTrend {
    fn spec(&self) -> FactorSpec {
        spec(self.output)
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.into()
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(ComputeState::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_requested(
            &[self.output.id().into()],
            data,
            &mut ComputeState::default(),
        )?
        .pop()
        .ok_or_else(|| err("Profit trend provider returned no output"))
    }
    fn compute_many(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(ids, data, &mut ComputeState::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(
            ids,
            data,
            state
                .downcast_mut::<ComputeState>()
                .ok_or_else(|| err("Profit trend provider state mismatch"))?,
        )
    }
}

struct Readers<'a> {
    income: FinancialPitReader<'a>,
    balance: Option<FinancialPitReader<'a>>,
    cash: Option<FinancialPitReader<'a>>,
    quarters: usize,
}

fn compute_requested(
    ids: &[String],
    data: &DataPool,
    state: &mut ComputeState,
) -> Result<Vec<FactorSeries>> {
    let requested: Vec<_> = ALL
        .into_iter()
        .filter(|o| ids.iter().any(|id| id == o.id()))
        .collect();
    if requested.is_empty() {
        return Ok(vec![]);
    }
    if requested != state.requested {
        state.snapshots = InstrumentAlignedSnapshotCache::default();
        state.requested = requested.clone();
    }
    let readers = Readers {
        income: data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?,
        balance: if requested.contains(&Output::Ocfa) {
            Some(data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )?)
        } else {
            None
        },
        cash: if requested.contains(&Output::Lpnp) {
            Some(data.financial_reader(
                DatasetId::StockCashFlow,
                ReportTypePreference::income_single_quarter(),
            )?)
        } else {
            None
        },
        quarters: requested
            .iter()
            .map(|o| o.quarters())
            .max()
            .unwrap_or(WINDOW),
    };
    let mut event_readers = vec![readers.income.clone()];
    event_readers.extend(readers.balance.iter().cloned());
    event_readers.extend(readers.cash.iter().cloned());
    let schedule = FinancialEventSchedule::from_pit_readers(&event_readers);
    let panel = data.stock_universe_panel()?;
    let n = panel.instruments().len();
    let mut values = vec![vec![None; panel.shape_len()]; requested.len()];
    let mut snapshots = vec![None; n];
    let mut qpt = vec![None; n];
    let mut last_date = None;
    let mut previous_day: Option<usize> = None;
    for (day, date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let changed = previous_day.is_some_and(|prev| {
            (0..n).any(|i| {
                panel.is_present_offset(day * n + i) != panel.is_present_offset(prev * n + i)
            })
        });
        if last_date.is_none() || changed || schedule.has_event_after_until(last_date, date) {
            snapshots = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.snapshots,
                &FinancialStockEventRoute::new(&event_readers.iter().collect::<Vec<_>>()),
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| readers.marker(code, date),
                |date, code, _| readers.snapshot(code, date, &requested),
            );
            if requested.contains(&Output::Qpt) {
                qpt = qpt_scores(&snapshots, panel.instruments());
            }
        }
        for (i, snapshot) in snapshots.iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                for (j, output) in requested.iter().enumerate() {
                    values[j][day * n + i] = if *output == Output::Qpt {
                        qpt[i]
                    } else {
                        snapshot.and_then(|s| s.raw[*output as usize])
                    };
                }
            }
        }
        last_date = Some(date);
        previous_day = Some(day);
    }
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    requested
        .into_iter()
        .zip(values)
        .map(|(output, values)| {
            let raw = panel.column_from_values(values)?;
            Ok(
                neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector)?
                    .to_factor_series(spec(output)),
            )
        })
        .collect()
}

fn quarter_ends(anchor: i32) -> Option<[i32; WINDOW + 1]> {
    let mut ends = [anchor; WINDOW + 1];
    for q in 1..ends.len() {
        ends[q] = previous_quarter_end_date(ends[q - 1])?;
    }
    Some(ends)
}

impl Readers<'_> {
    fn marker(&self, code: &str, date: i32) -> Option<FinancialEventMarker> {
        let ends = quarter_ends(self.income.latest_quarter_end_date(code, date)?)?;
        let mut marker = FinancialEventMarkerBuilder::new();
        for (q, end) in ends.into_iter().enumerate().take(self.quarters) {
            marker.include_reader_record_for_end_date(
                FinancialStatementDataset::Income,
                &self.income,
                code,
                date,
                end,
            );
            if q < WINDOW {
                if let Some(reader) = &self.balance {
                    marker.include_reader_record_for_end_date(
                        FinancialStatementDataset::BalanceSheet,
                        reader,
                        code,
                        date,
                        end,
                    );
                }
                if let Some(reader) = &self.cash {
                    marker.include_reader_record_for_end_date(
                        FinancialStatementDataset::CashFlow,
                        reader,
                        code,
                        date,
                        end,
                    );
                }
            }
        }
        marker.build()
    }

    fn snapshot(&self, code: &str, date: i32, requested: &[Output]) -> Option<Snapshot> {
        let ends = quarter_ends(self.income.latest_quarter_end_date(code, date)?)?;
        let income = std::array::from_fn::<_, { WINDOW + 1 }, _>(|q| {
            if q < self.quarters {
                self.income.record_for_end_date(code, date, ends[q])
            } else {
                None
            }
        });
        let get = |q: usize, name: &str| income[q].and_then(|r| clean(r.column(name)));
        let mut snapshot = Snapshot::default();
        for output in requested {
            match output {
                Output::Opmd => {
                    snapshot.raw[0] = opmd(
                        std::array::from_fn(|q| get(q, "operate_profit")),
                        std::array::from_fn(|q| get(q, "revenue")),
                    );
                }
                Output::Lpnp => {
                    let rows = std::array::from_fn(|q| {
                        let cash = self
                            .cash
                            .as_ref()?
                            .record_for_end_date(code, date, ends[q])?;
                        Some((
                            get(q, "n_income")?,
                            [
                                get(q, "non_oper_income")?,
                                clean(cash.column("c_paid_to_for_empl"))?,
                            ],
                        ))
                    });
                    snapshot.raw[1] = latest_ts_residual(rows);
                }
                Output::Ocfa => {
                    let rows = std::array::from_fn(|q| {
                        let balance = self
                            .balance
                            .as_ref()?
                            .record_for_end_date(code, date, ends[q])?;
                        Some((
                            get(q, "total_cogs")?,
                            [clean(balance.column("fix_assets"))?],
                        ))
                    });
                    snapshot.raw[2] = latest_ts_residual(rows);
                }
                Output::Qpt => {
                    let profits = std::array::from_fn(|q| get(q, "n_income_attr_p"));
                    (snapshot.growth, snapshot.acceleration) = qpt_metrics(profits);
                }
            }
        }
        Some(snapshot)
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn complete_sum(values: &[Option<f64>]) -> Option<f64> {
    clean(
        values
            .iter()
            .try_fold(0.0, |sum, value| Some(sum + clean(*value)?)),
    )
}

fn opmd(profit: [Option<f64>; 5], revenue: [Option<f64>; 5]) -> Option<f64> {
    let current_revenue = complete_sum(&revenue[..4]).filter(|v| *v > EPS)?;
    let prior_revenue = complete_sum(&revenue[1..]).filter(|v| *v > EPS)?;
    clean(Some(
        complete_sum(&profit[..4])? / current_revenue - complete_sum(&profit[1..])? / prior_revenue,
    ))
}

fn standardize(values: [f64; WINDOW]) -> Option<[f64; WINDOW]> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scale = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return None;
    }
    let scaled = values.map(|v| v / scale);
    let mean = scaled.iter().sum::<f64>() / WINDOW as f64;
    let centered = scaled.map(|v| v - mean);
    let sd = (centered.iter().map(|v| v * v).sum::<f64>() / WINDOW as f64).sqrt();
    (sd > EPS).then(|| centered.map(|v| v / sd))
}

fn dot(a: &[f64; WINDOW], b: &[f64; WINDOW]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

// Centering absorbs the intercept; reorthogonalized QR avoids normal-equation conditioning.
fn latest_ts_residual<const P: usize>(rows: [Option<(f64, [f64; P])>; WINDOW]) -> Option<f64> {
    let mut y = [0.0; WINDOW];
    let mut x = [[0.0; WINDOW]; P];
    for (q, row) in rows.into_iter().enumerate() {
        let (yi, xi) = row?;
        y[q] = clean(Some(yi))?;
        for p in 0..P {
            x[p][q] = clean(Some(xi[p]))?;
        }
    }
    let y = standardize(y)?;
    let mut basis = [[0.0; WINDOW]; P];
    let mut residual = y[0];
    for p in 0..P {
        let mut vector = standardize(x[p])?;
        for _ in 0..2 {
            for b in &basis[..p] {
                let projection = dot(&vector, b);
                for q in 0..WINDOW {
                    vector[q] -= projection * b[q];
                }
            }
        }
        let norm = dot(&vector, &vector).sqrt();
        if norm <= EPS {
            return None;
        }
        basis[p] = vector.map(|v| v / norm);
        residual -= dot(&basis[p], &y) * basis[p][0];
    }
    clean(Some(residual))
}

fn acceleration(profits: [Option<f64>; WINDOW]) -> Option<f64> {
    // For times 1..8, centered quadratic weights are orthogonal to intercept and time.
    let weights = [7.0, 1.0, -3.0, -5.0, -5.0, -3.0, 1.0, 7.0];
    let mut values = [0.0; WINDOW];
    for q in 0..WINDOW {
        values[q] = clean(profits[q])?;
    }
    let scale = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Some(0.0);
    }
    clean(Some(
        dot(&values.map(|v| v / scale), &weights) / 168.0 * scale,
    ))
}

fn qpt_metrics(profits: [Option<f64>; WINDOW + 1]) -> (Option<f64>, Option<f64>) {
    let growth = (|| {
        let current = complete_sum(&profits[..4])?;
        let previous = complete_sum(&profits[1..5]).filter(|v| v.abs() > EPS)?;
        clean(Some(current / previous - 1.0))
    })();
    let current = acceleration(std::array::from_fn(|q| profits[q]));
    let acceleration = current.or_else(|| acceleration(std::array::from_fn(|q| profits[q + 1])));
    (growth, acceleration)
}

fn qpt_scores(snapshots: &[Option<Snapshot>], instruments: &[String]) -> Vec<Option<f64>> {
    let mut scores = vec![None; snapshots.len()];
    let mut groups = vec![None; snapshots.len()];
    let mut valid = Vec::with_capacity(snapshots.len());
    for (i, snapshot) in snapshots.iter().enumerate() {
        if let Some(snapshot) = snapshot {
            if let Some(growth) = clean(snapshot.growth) {
                valid.push((i, growth));
            } else if clean(snapshot.acceleration).is_some() {
                groups[i] = Some(1);
            }
        }
    }
    // Tie-break by stock code so instrument reordering cannot change group membership.
    valid.sort_unstable_by(|(i, a), (j, b)| {
        a.total_cmp(b)
            .then_with(|| instruments[*i].cmp(&instruments[*j]))
    });
    for (rank, (i, _)) in valid.iter().enumerate() {
        groups[*i] = Some((rank * 3 / valid.len()).min(2));
    }
    for group in 0..3 {
        let mut members: Vec<_> = snapshots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if groups[i] == Some(group) {
                    Some((i, clean(s.as_ref()?.acceleration)?))
                } else {
                    None
                }
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        members.sort_unstable_by(|(i, _), (j, _)| instruments[*i].cmp(&instruments[*j]));
        let scale = members
            .iter()
            .map(|(_, a)| a.abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let mean = members.iter().map(|(_, a)| a / scale).sum::<f64>() / members.len() as f64;
        let sd = (members
            .iter()
            .map(|(_, a)| (a / scale - mean).powi(2))
            .sum::<f64>()
            / members.len() as f64)
            .sqrt();
        for (i, a) in members {
            let z = if sd > EPS {
                (a / scale - mean) / sd
            } else {
                0.0
            };
            scores[i] = clean(Some(group as f64 + 1.0 + z));
        }
    }
    scores
}

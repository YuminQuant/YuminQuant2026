use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::{add_months, analyst_fiscal_years};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, FinancialPitReader, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_fill_mean_by_group, cs_neutralize_regression, cs_zscore};

const LOOKBACK: usize = 110;
const EPS: f64 = 1e-12;
const PROFIT: &str = "n_income_attr_p";
const EQUITY: &str = "total_hldr_eqy_exc_min_int";
const FORECASTS: [&str; 4] = ["con_np_fy0", "con_np_fy1", "con_np_fy2", "con_np_fy3"];

pub struct StockDailyCgva;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCgva)
}

impl Factor for StockDailyCgva {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cgva".into(),
            aliases: vec!["CGVA".into()],
            name: "Consensus Growth Valuation Anomaly".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.1".into(),
            tags: ["CICC", "analyst", "fundamental", "consensus", "valuation", "pit",
                "monthly_update", "neutralize", "barra", "size", "sector", "daily"]
                .into_iter().map(str::to_string).collect(),
            description: "Month-end CGVA; hold FINAL neutralized cross-section by stock code daily. FY1=previous calendar year through Apr30 inclusive, current year from May1; FY0=FY1-1 and FY3=FY1+2. Map consensus FY0..3 to absolute years separately at each as-of date using the consensus PIT annual/May1 anchor; out-of-range historical years use only PIT-visible annual parent profit/10000. EEChange=(current FY1 NP-prior same absolute year NP)/abs(prior NP), prior date=last trading day on/before anchor minus three calendar months. Growth3=(FY3 NP/abs(FY0 NP))^(1/3)-1, Growth1=(FY1 NP-FY0 NP)/abs(FY0 NP); nonzero base and nonnegative FY3 required. Positive pe_ttm regressed on Growth3, PIT parent profit TTM/average parent equity at q and q-4 (both positive), ln(total_mv), SW L1. All-market valid EEChange split into three ascending equal-count groups, stock code breaks ties. Within each group population zscores of PE residual and Growth3-Growth1; zero deviation null. Raw=group score-Z(PE residual)+Z(term). Fill each of the three score columns with its SW L1 industry mean BEFORE summation; an unsupported industry remains null. Then SW L1/Barra SIZE neutralize, without post-composite fill, clips or extra zscore. Excludes BJ; no PV anchor. Projected input columns and sparse month-end/three-month comparison dates; state keeps only last final cross-section.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockConsensus, &FORECASTS),
                DataRequest::new(DatasetId::StockDailyBasic, &["pe_ttm", "total_mv"]),
                DataRequest::financial_quarters(DatasetId::StockIncome, &[PROFIT], 12),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &[EQUITY], 12),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: LOOKBACK },
        }
    }

    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        let anchors = candidate_anchors(&context.load_dates);
        let mut forecast_dates = anchors.clone();
        for &anchor in &anchors {
            let end = context
                .load_dates
                .partition_point(|d| *d <= add_months(anchor, -3));
            if end > 0 {
                forecast_dates.insert(context.load_dates[end - 1]);
            }
        }
        self.spec()
            .dependencies
            .into_iter()
            .map(|request| match request.dataset {
                DatasetId::StockConsensus => {
                    request.with_explicit_dates(forecast_dates.iter().copied().collect())
                }
                DatasetId::StockDailyBasic | DatasetId::StockBarraDaily => {
                    request.with_explicit_dates(anchors.iter().copied().collect())
                }
                _ => request,
            })
            .collect()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<State>::default()
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(data, &mut State::default())
    }

    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "cgva") {
            return Ok(vec![]);
        }
        let state = state
            .downcast_mut::<State>()
            .ok_or_else(|| err("cgva state type mismatch"))?;
        Ok(vec![compute(data, state)?])
    }
}

#[derive(Default)]
struct State {
    last_anchor: Option<i32>,
    last_target: Option<i32>,
    final_values: HashMap<String, Option<f64>>,
}

fn candidate_anchors(dates: &[i32]) -> BTreeSet<i32> {
    dates
        .windows(2)
        .filter(|pair| pair[0] / 100 != pair[1] / 100)
        .map(|pair| pair[0])
        .chain(dates.last().copied())
        .collect()
}

struct Rows<const N: usize> {
    values: HashMap<(i32, usize), [Option<f64>; N]>,
}

impl<const N: usize> Rows<N> {
    fn new(
        table: &Table,
        columns: &[&str; N],
        dates: &BTreeSet<i32>,
        codes: &[String],
    ) -> Result<Self> {
        let stocks: HashMap<_, _> = codes
            .iter()
            .enumerate()
            .map(|(i, code)| (code.as_str(), i))
            .collect();
        let ts = table.required_utf8("ts_code")?;
        let ds = table.required_i32_date_cast("trade_date")?;
        let selected: Vec<_> = (0..table.len)
            .filter_map(|row| {
                let date = ds[row]?;
                dates.contains(&date).then_some(())?;
                Some((row, (date, *stocks.get(ts[row].as_deref()?)?)))
            })
            .collect();
        let mut values: HashMap<_, _> = selected.iter().map(|(_, key)| (*key, [None; N])).collect();
        for (j, column) in columns.iter().enumerate() {
            let source = table.required_f64_cast(column)?;
            for &(row, key) in &selected {
                values.get_mut(&key).unwrap()[j] = source[row].filter(|v| v.is_finite());
            }
        }
        Ok(Self { values })
    }

    fn get(&self, date: i32, stock: usize, column: usize) -> Option<f64> {
        self.values
            .get(&(date, stock))?
            .get(column)
            .copied()
            .flatten()
    }
}

fn fy1(date: i32) -> i32 {
    date / 10000 - i32::from(date % 10000 <= 430)
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn revision(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    let current = current.filter(|v| v.is_finite())?;
    let previous = previous.filter(|v| v.is_finite() && v.abs() > EPS)?;
    finite((current - previous) / previous.abs())
}

fn growth(base: Option<f64>, next: Option<f64>, third: Option<f64>) -> Option<(f64, f64)> {
    let base = base.filter(|v| v.is_finite() && v.abs() > EPS)?;
    let next = next.filter(|v| v.is_finite())?;
    let third = third.filter(|v| v.is_finite() && *v >= 0.0)?;
    let long = finite((third / base.abs()).cbrt() - 1.0)?;
    let short = finite((next - base) / base.abs())?;
    Some((long, finite(long - short)?))
}

fn profit_for_year(
    np: &Rows<4>,
    annual: &FinancialPitReader<'_>,
    code: &str,
    stock: usize,
    date: i32,
    year: i32,
) -> Option<f64> {
    let source = analyst_fiscal_years(date, code, annual)[0];
    if let Ok(column @ 0..=3) = usize::try_from(year - source) {
        return np.get(date, stock, column);
    }
    finite(
        annual
            .record_for_end_date(code, date, year * 10000 + 1231)?
            .column(PROFIT)?
            / 10000.0,
    )
}

fn roe(
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<f64> {
    let end = income.latest_quarter_end_date(code, date)?;
    let current = balance
        .record_for_end_date(code, date, end)?
        .column(EQUITY)
        .filter(|v| v.is_finite() && *v > 0.0)?;
    let previous = balance
        .record_for_end_date(code, date, end - 10000)?
        .column(EQUITY)
        .filter(|v| v.is_finite() && *v > 0.0)?;
    finite(income.ttm_sum_for_end_date(code, date, end, PROFIT)? / (0.5 * current + 0.5 * previous))
}

fn grouped_score(
    ee: &[Option<f64>],
    bias: &[Option<f64>],
    term: &[Option<f64>],
    codes: &[String],
    industries: &[Option<String>],
) -> Vec<Option<f64>> {
    let mut order: Vec<_> = ee
        .iter()
        .enumerate()
        .filter_map(|(i, value)| value.filter(|v| v.is_finite()).map(|v| (i, v)))
        .collect();
    order.sort_unstable_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| codes[a.0].cmp(&codes[b.0]))
    });
    let mut members = [vec![], vec![], vec![]];
    for (rank, &(i, _)) in order.iter().enumerate() {
        members[rank * 3 / order.len()].push(i);
    }
    let mut scores = vec![vec![None; ee.len()]; 3];
    for (group, indices) in members.iter().enumerate() {
        let value: Vec<_> = indices
            .iter()
            .map(|&i| bias[i].filter(|v| v.is_finite()))
            .collect();
        let structure: Vec<_> = indices
            .iter()
            .map(|&i| term[i].filter(|v| v.is_finite()))
            .collect();
        let zv = cs_zscore(&value);
        let zt = cs_zscore(&structure);
        for (j, &i) in indices.iter().enumerate() {
            scores[0][i] = Some(group as f64 + 1.0);
            scores[1][i] = zv[j].map(|v| -v);
            scores[2][i] = zt[j];
        }
    }
    // EE-group zscores are not industry-neutral: use actual industry means.
    let scores: Vec<_> = scores
        .iter()
        .map(|s| cs_fill_mean_by_group(s, industries))
        .collect();
    (0..ee.len())
        .map(|i| finite(scores[0][i]? + scores[1][i]? + scores[2][i]?))
        .collect()
}

fn compute(data: &DataPool, state: &mut State) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let calendar = data.trading_calendar()?;
    let codes = panel.instruments();
    let n = codes.len();
    let anchors: BTreeSet<_> = panel
        .dates()
        .iter()
        .filter(|d| panel.is_target_date(**d))
        .filter_map(|&d| calendar.month_end_on_or_before(d))
        .collect();
    let mut forecast_dates = anchors.clone();
    let mut prior_dates = HashMap::new();
    for &anchor in &anchors {
        let previous = calendar
            .last_open_on_or_before(add_months(anchor, -3))
            .ok_or_else(|| err("cgva three-month comparison date unavailable"))?;
        if panel.dates().binary_search(&previous).is_err() {
            return Err(err(
                "cgva three-month comparison date missing from warmup panel",
            ));
        }
        forecast_dates.insert(previous);
        prior_dates.insert(anchor, previous);
    }
    let np = Rows::new(
        data.daily(DatasetId::StockConsensus)?,
        &FORECASTS,
        &forecast_dates,
        codes,
    )?;
    let basic = Rows::new(
        data.daily(DatasetId::StockDailyBasic)?,
        &["pe_ttm", "total_mv"],
        &anchors,
        codes,
    )?;
    let size = Rows::new(
        data.daily(DatasetId::StockBarraDaily)?,
        &["SIZE"],
        &anchors,
        codes,
    )?;
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let annual =
        data.financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel
        .dates()
        .iter()
        .enumerate()
        .filter(|(_, d)| panel.is_target_date(**d))
    {
        if state.last_target.is_some_and(|previous| date < previous) {
            return Err(err("cgva provider state cannot move backwards"));
        }
        if let Some(anchor) = calendar.month_end_on_or_before(date) {
            if state.last_anchor != Some(anchor) {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("cgva month-end missing from warmup panel"))?;
                let previous = prior_dates[&anchor];
                let year = fy1(anchor);
                let mut groups = sectors.groups_for(anchor, codes);
                let mut ee = vec![None; n];
                let mut long = vec![None; n];
                let mut term = vec![None; n];
                let mut pe = vec![None; n];
                let mut profitability = vec![None; n];
                let mut log_mv = vec![None; n];
                let mut exposure = vec![None; n];
                for (i, code) in codes.iter().enumerate() {
                    if code.ends_with(".BJ") || !panel.is_present_offset(anchor_day * n + i) {
                        groups[i] = None;
                        continue;
                    }
                    let current = profit_for_year(&np, &annual, code, i, anchor, year);
                    let old = profit_for_year(&np, &annual, code, i, previous, year);
                    ee[i] = revision(current, old);
                    let base = profit_for_year(&np, &annual, code, i, anchor, year - 1);
                    let third = profit_for_year(&np, &annual, code, i, anchor, year + 2);
                    if let Some((g, t)) = growth(base, current, third) {
                        long[i] = Some(g);
                        term[i] = Some(t);
                    }
                    pe[i] = basic.get(anchor, i, 0).filter(|v| *v > 0.0);
                    profitability[i] = roe(&income, &balance, code, anchor);
                    log_mv[i] = basic
                        .get(anchor, i, 1)
                        .filter(|v| *v > 0.0)
                        .and_then(|v| finite(v.ln()));
                    exposure[i] = size.get(anchor, i, 0);
                }
                let bias = cs_neutralize_regression(
                    &pe,
                    &[&long, &profitability, &log_mv],
                    Some(&groups),
                    None,
                );
                let raw = grouped_score(&ee, &bias, &term, codes, &groups);
                let final_values =
                    cs_neutralize_regression(&raw, &[&exposure], Some(&groups), None);
                state.final_values = codes
                    .iter()
                    .cloned()
                    .zip(
                        final_values
                            .into_iter()
                            .map(|v| v.filter(|x| x.is_finite())),
                    )
                    .collect();
                state.last_anchor = Some(anchor);
            }
        }
        for (i, code) in codes.iter().enumerate() {
            if !code.ends_with(".BJ") && panel.is_present_offset(day * n + i) {
                values[day * n + i] = state.final_values.get(code).copied().flatten();
            }
        }
        state.last_target = Some(date);
    }
    Ok(panel
        .column_from_values(values)?
        .to_factor_series(StockDailyCgva.spec()))
}

use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::add_months;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_zscore};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    DirectReturn,
    RawReversal,
    Turnover,
    Volatility,
    IndirectReturn,
    MultilayerReturn,
    DirectReversal,
    IndirectReversal,
    MultilayerReversal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Output(pub Kind, pub usize);

impl Output {
    pub fn id(self) -> String {
        let prefix = match self.0 {
            Kind::DirectReturn => "analyst_cf1_ret",
            Kind::RawReversal => "analyst_cf_rev_raw",
            Kind::Turnover => "analyst_cf_turn",
            Kind::Volatility => "analyst_cf_std",
            Kind::IndirectReturn => "analyst_cf2_ret",
            Kind::MultilayerReturn => "analyst_cf_ret",
            Kind::DirectReversal => "analyst_cf1_rev",
            Kind::IndirectReversal => "analyst_cf2_rev",
            Kind::MultilayerReversal => "analyst_cf_rev",
        };
        format!("{prefix}_{}", self.1)
    }

    fn indirect(self) -> bool {
        matches!(
            self.0,
            Kind::IndirectReturn
                | Kind::MultilayerReturn
                | Kind::IndirectReversal
                | Kind::MultilayerReversal
        )
    }

    fn metric(self) -> Metric {
        match self.0 {
            Kind::Turnover => Metric::Turnover,
            Kind::Volatility => Metric::Volatility,
            _ => Metric::Return,
        }
    }
}

pub const OUTPUTS: [Output; 9] = [
    Output(Kind::DirectReturn, 20),
    Output(Kind::RawReversal, 20),
    Output(Kind::Turnover, 20),
    Output(Kind::Volatility, 20),
    Output(Kind::IndirectReturn, 20),
    Output(Kind::MultilayerReturn, 20),
    Output(Kind::DirectReversal, 20),
    Output(Kind::IndirectReversal, 20),
    Output(Kind::MultilayerReversal, 20),
];

pub struct CoverageNetwork(pub Output);

impl Factor for CoverageNetwork {
    fn spec(&self) -> FactorSpec {
        let output = self.0;
        let mut dependencies = vec![
            DataRequest::new(
                DatasetId::StockAnalystReport,
                &["report_date", "org_name", "author_name"],
            ),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]).with_panel_only(),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ];
        if output.0 == Kind::Turnover {
            dependencies.push(
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"])
                    .with_panel_only(),
            );
        } else {
            dependencies.extend([
                DataRequest::new(DatasetId::StockDailyPv, &["close"]).with_panel_only(),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]).with_panel_only(),
            ]);
        }
        let formula = match output.0 {
            Kind::DirectReturn => "CF1_RET=direct log1p(co-analyst-count) weighted return; also the July-2022 CF_RET",
            Kind::RawReversal => "July-2022 CF_REV=CF1_RET-own return (raw difference, without internal zscore)",
            Kind::Turnover => "CF_TURN=direct weighted mean turnover-own mean turnover",
            Kind::Volatility => "CF_STD=direct weighted daily-return population volatility-own volatility",
            Kind::IndirectReturn => "CF2_RET=indirect weighted return",
            Kind::MultilayerReturn => "CF_RET=Z(CF1_RET)+Z(CF2_RET); confirmed adaptation replacing the report's cancelling adjusted-indirect formula",
            Kind::DirectReversal => "October-2022 CF1_REV=Z(CF1_RET)-Z(own return)",
            Kind::IndirectReversal => "CF2_REV=Z(CF2_RET)-Z(own return); uses unadjusted indirect return",
            Kind::MultilayerReversal => "CF_REV=[Z(CF1_RET)-Z(own return)]+[Z(CF2_RET)-Z(own return)]",
        };
        FactorSpec {
            id: output.id(), aliases: vec![output.id().to_ascii_uppercase()], name: output.id(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["HTZQ", "analyst", "price_volume", "cs_network", "daily", "quarterly_network", "neutralize", "size", "sector"].into_iter().map(str::to_string).collect(),
            description: format!("{formula}. Window={} trading sessions, min_periods=1. Adjusted close=close*adj_factor; simple returns between adjacent calendar sessions only; cumulative return compounds finite daily returns, missing days skipped, no valid returns is null. Turnover uses mean turnover_rate_f (percent), volatility uses population std. Latest completed trading quarter end (inclusive) builds coverage from reports in (snapshot minus six calendar months,snapshot]. Analyst identity institution plus individual split author; deduplicate analyst-stock membership across reports and forecast years. Direct A_ij=ln(1+n_ij), zero diagonal; indirect M=A*A with diagonal and every direct edge masked to zero. Current missing peer values excluded with weight renormalization. No dense NxN graph, no indirect matrix; sparse two-hop products subtract direct/self contributions. Cache one quarterly graph keyed by stock code, no disk intermediate. Main grid=stock-basic universe, excludes BJ and stocks not listed at the graph snapshot. Graph frozen within quarter; signals and SW L1/Barra SIZE residual recomputed daily. Internal zscores only where formula requires; no final zscore, winsorization or missing fill. Base return/turnover/volatility and adjusted-indirect are not outputs. Only requested outputs, metrics and windows calculated.", output.1),
            dependencies, intraday_raw_dependencies: vec![],
            // Covers every possible prior quarter boundary; market inputs are projected to the required window.
            lookback: Lookback { trading_days: 93 },
        }
    }

    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        let first = context
            .target_dates
            .first()
            .copied()
            .unwrap_or(context.start_date);
        let position = context.load_dates.partition_point(|date| *date < first);
        let market_dates = context.load_dates[position.saturating_sub(self.0 .1)..].to_vec();
        self.spec()
            .dependencies
            .into_iter()
            .map(|request| {
                let dates = match request.dataset {
                    DatasetId::StockBarraDaily => context.target_dates.clone(),
                    DatasetId::StockDailyPv
                    | DatasetId::StockAdjFactor
                    | DatasetId::StockDailyBasic => market_dates.clone(),
                    _ => context.load_dates.clone(),
                };
                request.with_explicit_dates(dates)
            })
            .collect()
    }

    fn compute_provider_key(&self) -> String {
        "htzq_coverage_network".into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id()], context, data, &mut State::default())?
            .pop()
            .ok_or_else(|| err("missing analyst coverage network output"))
    }

    fn compute_many(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute(ids, context, data, &mut State::default())
    }

    fn compute_many_stateful(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute(
            ids,
            context,
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("analyst coverage network state mismatch"))?,
        )
    }
}

#[derive(Default)]
struct State {
    graph: Option<Graph>,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    node: usize,
    weight: f64,
}

struct Graph {
    snapshot: i32,
    codes: Vec<String>,
    covered: Vec<bool>,
    direct: Vec<Vec<Edge>>,
    exclusion: Option<Vec<Vec<Edge>>>,
}

fn quarter_key(date: i32) -> (i32, i32) {
    (date / 10000, ((date / 100) % 100 - 1) / 3)
}

fn snapshot_date(calendar: &TradingCalendar, date: i32) -> Result<i32> {
    let (year, quarter) = quarter_key(date);
    let month = 3 * (quarter + 1);
    let end = year * 10000 + month * 100 + if month == 6 || month == 9 { 30 } else { 31 };
    if let Some(last) = calendar.last_open_on_or_before(end) {
        if last <= date
            && calendar
                .open_date_after(last, 1)
                .is_some_and(|next| quarter_key(next) != quarter_key(last))
        {
            return Ok(last);
        }
    }
    let previous = if quarter == 0 {
        (year - 1) * 10000 + 1231
    } else {
        let month = quarter * 3;
        year * 10000 + month * 100 + if month == 6 || month == 9 { 30 } else { 31 }
    };
    calendar
        .last_open_on_or_before(previous)
        .ok_or_else(|| err("no completed quarter for analyst coverage network"))
}

fn build_graph(data: &DataPool, panel: &DailyPanel, snapshot: i32) -> Result<Graph> {
    let day = panel
        .dates()
        .binary_search(&snapshot)
        .map_err(|_| err("analyst network snapshot is outside stock universe warmup"))?;
    let mut codes: Vec<_> = panel
        .instruments()
        .iter()
        .enumerate()
        .filter(|(stock, code)| {
            !is_bj_stock(code) && panel.is_present_offset(day * panel.instruments().len() + stock)
        })
        .map(|(_, code)| code.clone())
        .collect();
    codes.sort();
    let lookup: HashMap<_, _> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (code.as_str(), i))
        .collect();
    let reader = data.analyst_report_reader()?;
    let stocks = reader.required_utf8("ts_code")?;
    let dates = reader.required_i32_date_cast("report_date")?;
    let orgs = reader.required_utf8("org_name")?;
    let authors = reader.required_utf8("author_name")?;
    let cutoff = add_months(snapshot, -6);
    let mut groups: HashMap<(&str, &str), BTreeSet<usize>> = HashMap::new();
    for row in 0..reader.len {
        let (Some(date), Some(code), Some(org), Some(author)) = (
            dates[row],
            stocks[row].as_deref(),
            orgs[row].as_deref(),
            authors[row].as_deref(),
        ) else {
            continue;
        };
        if date <= cutoff || date > snapshot || org.trim().is_empty() {
            continue;
        }
        let Some(&stock) = lookup.get(code) else {
            continue;
        };
        for name in author
            .split([',', '，', '、', ';', '；', '/', '|'])
            .map(str::trim)
        {
            if name.is_empty() || ["nan", "None", "--", "佚名"].contains(&name) {
                continue;
            }
            groups.entry((org.trim(), name)).or_default().insert(stock);
        }
    }
    let memberships: Vec<Vec<_>> = groups
        .into_values()
        .map(|stocks| stocks.into_iter().collect())
        .collect();
    let mut stock_groups = vec![Vec::new(); codes.len()];
    for (group, stocks) in memberships.iter().enumerate() {
        for &stock in stocks {
            stock_groups[stock].push(group);
        }
    }
    let covered = stock_groups
        .iter()
        .map(|groups| !groups.is_empty())
        .collect();
    // Per-worker counts are reused across rows; only actual edges survive the build.
    let direct = (0..codes.len())
        .into_par_iter()
        .map_init(
            || (vec![0u32; codes.len()], Vec::<usize>::new()),
            |(counts, touched), stock| {
                for &group in &stock_groups[stock] {
                    for &peer in &memberships[group] {
                        if peer == stock {
                            continue;
                        }
                        if counts[peer] == 0 {
                            touched.push(peer);
                        }
                        counts[peer] += 1;
                    }
                }
                touched.sort_unstable();
                touched
                    .drain(..)
                    .map(|peer| {
                        let weight = f64::from(counts[peer]).ln_1p();
                        counts[peer] = 0;
                        Edge { node: peer, weight }
                    })
                    .collect()
            },
        )
        .collect();
    Ok(Graph {
        snapshot,
        codes,
        covered,
        direct,
        exclusion: None,
    })
}

fn sparse_dot(left: &[Edge], right: &[Edge]) -> f64 {
    let (mut i, mut j, mut sum) = (0, 0, 0.0);
    while i < left.len() && j < right.len() {
        match left[i].node.cmp(&right[j].node) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += left[i].weight * right[j].weight;
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

impl Graph {
    fn prepare_indirect(&mut self) {
        if self.exclusion.is_some() {
            return;
        }
        // Mx=A(Ax)-Bx, with B restricted to the diagonal and direct-edge support.
        // This is exactly the report's masked A^2 without storing its dense complement.
        self.exclusion = Some(
            self.direct
                .par_iter()
                .enumerate()
                .map(|(stock, edges)| {
                    let mut excluded = Vec::with_capacity(edges.len() + 1);
                    let diagonal = sparse_dot(edges, edges);
                    if diagonal > 0.0 {
                        excluded.push(Edge {
                            node: stock,
                            weight: diagonal,
                        });
                    }
                    for edge in edges {
                        let weight = sparse_dot(edges, &self.direct[edge.node]);
                        if weight > 0.0 {
                            excluded.push(Edge {
                                node: edge.node,
                                weight,
                            });
                        }
                    }
                    excluded.sort_unstable_by_key(|edge| edge.node);
                    excluded
                })
                .collect(),
        );
    }

    fn moments(&self, values: &[Option<f64>]) -> Vec<[f64; 2]> {
        self.direct
            .par_iter()
            .map(|edges| {
                let mut sums = [0.0; 2];
                for edge in edges {
                    if let Some(value) = values[edge.node] {
                        sums[0] += edge.weight * value;
                        sums[1] += edge.weight;
                    }
                }
                sums
            })
            .collect()
    }

    fn indirect_mean(&self, values: &[Option<f64>], direct: &[[f64; 2]]) -> Vec<Option<f64>> {
        let exclusion = self.exclusion.as_ref().expect("indirect support prepared");
        self.direct
            .par_iter()
            .enumerate()
            .map(|(stock, edges)| {
                let mut total = [0.0; 2];
                for edge in edges {
                    total[0] += edge.weight * direct[edge.node][0];
                    total[1] += edge.weight * direct[edge.node][1];
                }
                let scale = total[1];
                for edge in &exclusion[stock] {
                    if let Some(value) = values[edge.node] {
                        total[0] -= edge.weight * value;
                        total[1] -= edge.weight;
                    }
                }
                // Pure direct cliques have zero indirect support; discard roundoff ghosts.
                if total[1] <= scale.max(1.0) * 1e-12 {
                    None
                } else {
                    finite(total[0] / total[1])
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Metric {
    Return,
    Turnover,
    Volatility,
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn metric_value(
    metric: Metric,
    window: usize,
    day: usize,
    stock: usize,
    width: usize,
    prices: Option<&PanelColumn>,
    turnover: Option<&PanelColumn>,
) -> Option<f64> {
    if metric != Metric::Turnover {
        prices?.values()[day * width + stock]?;
    }
    let (mut count, mut sum, mut mean, mut m2) = (0usize, 0.0, 0.0, 0.0);
    for u in (day + 1).saturating_sub(window)..=day {
        let value = if metric == Metric::Turnover {
            turnover?.values()[u * width + stock].filter(|v| v.is_finite() && *v >= 0.0)
        } else if u > 0 {
            let current = prices?.values()[u * width + stock];
            let previous = prices?.values()[(u - 1) * width + stock];
            current.zip(previous).and_then(|(a, b)| finite(a / b - 1.0))
        } else {
            None
        };
        let Some(value) = value else { continue };
        count += 1;
        if metric == Metric::Return {
            sum += value.ln_1p();
        } else {
            sum += value;
        }
        let delta = value - mean;
        mean += delta / count as f64;
        m2 += delta * (value - mean);
    }
    if count == 0 {
        return None;
    }
    finite(match metric {
        Metric::Return => sum.exp_m1(),
        Metric::Turnover => sum / count as f64,
        Metric::Volatility => (m2.max(0.0) / count as f64).sqrt(),
    })
}

fn difference(left: &[Option<f64>], right: &[Option<f64>]) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(a, b)| a.zip(*b).and_then(|(a, b)| finite(a - b)))
        .collect()
}

fn sum_pair(left: &[Option<f64>], right: &[Option<f64>]) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(a, b)| a.zip(*b).and_then(|(a, b)| finite(a + b)))
        .collect()
}

fn mask_inactive(values: &mut [Option<f64>], active: &[bool]) {
    for (value, active) in values.iter_mut().zip(active) {
        if !active {
            *value = None;
        }
    }
}

fn compute(
    ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut State,
) -> Result<Vec<FactorSeries>> {
    let requested: Vec<_> = OUTPUTS
        .into_iter()
        .filter(|output| ids.contains(&output.id()))
        .collect();
    if requested.len() != ids.len() {
        return Err(err("unknown or duplicate analyst coverage network output"));
    }
    if requested.is_empty() || context.target_dates.is_empty() {
        return Ok(Vec::new());
    }
    let panel = data.stock_universe_panel()?;
    let width = panel.instruments().len();
    let prices = if requested
        .iter()
        .any(|output| output.metric() != Metric::Turnover)
    {
        let close = panel.column_from_panel(data.daily_panel(DatasetId::StockDailyPv)?, "close")?;
        let adj =
            panel.column_from_panel(data.daily_panel(DatasetId::StockAdjFactor)?, "adj_factor")?;
        Some(close.zip_binary(&adj, |close, adj| {
            close.zip(adj).and_then(|(close, adj)| {
                if close > 0.0 && adj > 0.0 {
                    finite(close * adj)
                } else {
                    None
                }
            })
        })?)
    } else {
        None
    };
    let turnover = if requested
        .iter()
        .any(|output| output.metric() == Metric::Turnover)
    {
        Some(panel.column_from_panel(
            data.daily_panel(DatasetId::StockDailyBasic)?,
            "turnover_rate_f",
        )?)
    } else {
        None
    };
    let size = panel.column_from_panel(data.daily_panel(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let lookup: HashMap<_, _> = panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(i, code)| (code.as_str(), i))
        .collect();
    let metrics: BTreeSet<_> = requested
        .iter()
        .map(|output| (output.metric(), output.1))
        .collect();
    let mut series: Vec<_> = requested
        .iter()
        .map(|output| FactorSeries {
            spec: CoverageNetwork(*output).spec(),
            values: Vec::new(),
        })
        .collect();
    let needs_indirect = requested.iter().any(|output| output.indirect());
    for &date in &context.target_dates {
        let day = panel
            .dates()
            .binary_search(&date)
            .map_err(|_| err("missing analyst network target date in universe"))?;
        let snapshot = snapshot_date(data.trading_calendar()?, date)?;
        if state
            .graph
            .as_ref()
            .is_none_or(|graph| graph.snapshot != snapshot)
        {
            state.graph = Some(build_graph(data, panel, snapshot)?);
        }
        let graph = state.graph.as_mut().expect("network built");
        if needs_indirect {
            graph.prepare_indirect();
        }
        let positions: Vec<_> = graph
            .codes
            .iter()
            .map(|code| lookup.get(code.as_str()).copied())
            .collect();
        let active: Vec<_> = positions
            .iter()
            .enumerate()
            .map(|(node, stock)| {
                graph.covered[node]
                    && stock.is_some_and(|stock| panel.is_present_offset(day * width + stock))
            })
            .collect();
        let groups = sector.groups_for(date, panel.instruments());
        for &(metric, window) in &metrics {
            let own: Vec<_> = positions
                .par_iter()
                .enumerate()
                .map(|(node, stock)| {
                    if !active[node] {
                        return None;
                    }
                    let stock = (*stock)?;
                    metric_value(
                        metric,
                        window,
                        day,
                        stock,
                        width,
                        prices.as_ref(),
                        turnover.as_ref(),
                    )
                })
                .collect();
            let moments = graph.moments(&own);
            let mut direct: Vec<_> = moments
                .iter()
                .map(|s| {
                    if s[1] > 0.0 {
                        finite(s[0] / s[1])
                    } else {
                        None
                    }
                })
                .collect();
            let mut indirect = if requested
                .iter()
                .any(|output| output.metric() == metric && output.1 == window && output.indirect())
            {
                graph.indirect_mean(&own, &moments)
            } else {
                Vec::new()
            };
            // Keep all graph intermediaries, but not absent targets in CS statistics.
            mask_inactive(&mut direct, &active);
            mask_inactive(&mut indirect, &active);
            let need_z = requested.iter().any(|output| {
                output.metric() == metric
                    && output.1 == window
                    && matches!(
                        output.0,
                        Kind::MultilayerReturn
                            | Kind::DirectReversal
                            | Kind::IndirectReversal
                            | Kind::MultilayerReversal
                    )
            });
            let (z_own, z_direct, z_indirect) = if need_z {
                (cs_zscore(&own), cs_zscore(&direct), cs_zscore(&indirect))
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            for (output, result) in requested
                .iter()
                .zip(&mut series)
                .filter(|(output, _)| output.metric() == metric && output.1 == window)
            {
                let mut raw = vec![None; width];
                let values = match output.0 {
                    Kind::DirectReturn => direct.clone(),
                    Kind::RawReversal | Kind::Turnover | Kind::Volatility => {
                        difference(&direct, &own)
                    }
                    Kind::IndirectReturn => indirect.clone(),
                    Kind::MultilayerReturn => sum_pair(&z_direct, &z_indirect),
                    Kind::DirectReversal => difference(&z_direct, &z_own),
                    Kind::IndirectReversal => difference(&z_indirect, &z_own),
                    Kind::MultilayerReversal => sum_pair(
                        &difference(&z_direct, &z_own),
                        &difference(&z_indirect, &z_own),
                    ),
                };
                for (node, stock) in positions.iter().enumerate() {
                    if let Some(stock) =
                        stock.filter(|stock| panel.is_present_offset(day * width + stock))
                    {
                        raw[stock] = values[node];
                    }
                }
                let residual = cs_neutralize_regression(
                    &raw,
                    &[&size.values()[day * width..(day + 1) * width]],
                    Some(&groups),
                    None,
                );
                for (stock, code) in panel.instruments().iter().enumerate() {
                    if !is_bj_stock(code) && panel.is_present_offset(day * width + stock) {
                        result.values.push(FactorValue {
                            key: FactorRowKey::Daily {
                                trade_date: date,
                                ts_code: code.clone(),
                            },
                            value: residual[stock].filter(|v| v.is_finite()),
                        });
                    }
                }
            }
        }
    }
    Ok(series)
}

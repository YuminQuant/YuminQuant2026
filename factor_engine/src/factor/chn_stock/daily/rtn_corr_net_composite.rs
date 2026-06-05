use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;
use crate::operators::cs_zscore;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const TOP_EDGE_FRACTION: f64 = 0.03;
const MAX_LANDMARKS: usize = 32;
const DIST_EPS: f64 = 1e-12;

pub struct StockDailyRtnCorrNetComposite;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRtnCorrNetComposite)
}

impl Factor for StockDailyRtnCorrNetComposite {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rtn_corr_net_composite".to_string(),
            aliases: vec![
                "RtnCorrNetComposite".to_string(),
                "ReturnCorrelationNetworkComposite".to_string(),
            ],
            name: "rtn_corr_net_composite".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "CJZQ return-correlation network composite factor. It uses a strict 20-day log-return window, keeps the top 3% absolute Pearson-correlation pairs, combines landmark harmonic centrality and Burt constraint, and neutralizes the composite by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
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
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let returns = close.zip_binary(&pre_close, log_return)?;
        let eligible = eligible_instruments(&panel);
        let values = rtn_corr_net_composite_values(&panel, returns.values(), &eligible);
        let raw = panel.column_from_values(values)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "CJZQ",
        "cs_network",
        "correlation",
        "harmonic",
        "constraint",
        "network",
        "return",
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

fn log_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    let (Some(close), Some(pre_close)) = (clean(close), clean(pre_close)) else {
        return None;
    };
    if close <= f64::EPSILON || pre_close <= f64::EPSILON {
        return None;
    }
    let value = (close / pre_close).ln();
    value.is_finite().then_some(value)
}

fn eligible_instruments(panel: &DailyPanel) -> Vec<bool> {
    panel
        .instruments()
        .iter()
        .map(|ts_code| !is_bj_stock(ts_code))
        .collect()
}

fn rtn_corr_net_composite_values(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let mut harmonic_values = vec![None; panel.shape_len()];
    let mut constraint_values = vec![None; panel.shape_len()];

    for date_idx in WINDOW - 1..panel.dates().len() {
        let Some(window) =
            strict_standardized_window(returns, eligible, instrument_count, date_idx)
        else {
            continue;
        };
        if window.codes.len() < 2 {
            continue;
        }
        let edges = top_abs_correlation_edges(&window.vectors, TOP_EDGE_FRACTION);
        if edges.is_empty() {
            continue;
        }
        let adjacency = adjacency_list(window.codes.len(), &edges);
        let harmonic = landmark_harmonic_centrality(window.codes.len(), &adjacency, MAX_LANDMARKS);
        let constraint = constraint_centrality(window.codes.len(), &adjacency);
        let offset = date_idx * instrument_count;
        for (local_idx, code_idx) in window.codes.iter().enumerate() {
            harmonic_values[offset + code_idx] = harmonic[local_idx];
            constraint_values[offset + code_idx] = constraint[local_idx];
        }
    }

    let harmonic = panel
        .column_from_values(harmonic_values)
        .and_then(|column| column.cs(cs_zscore));
    let constraint = panel
        .column_from_values(constraint_values)
        .and_then(|column| column.cs(cs_zscore));
    match (harmonic, constraint) {
        (Ok(harmonic), Ok(constraint)) => harmonic
            .zip_binary(&constraint, |harmonic, constraint| {
                match (clean(harmonic), clean(constraint)) {
                    (Some(harmonic), Some(constraint)) => {
                        let value = 0.5 * harmonic - 0.5 * constraint;
                        value.is_finite().then_some(value)
                    }
                    _ => None,
                }
            })
            .map(|column| column.values().to_vec())
            .unwrap_or_else(|_| vec![None; panel.shape_len()]),
        _ => vec![None; panel.shape_len()],
    }
}

struct StandardizedWindow {
    codes: Vec<usize>,
    vectors: Vec<[f64; WINDOW]>,
}

fn strict_standardized_window(
    returns: &[Option<f64>],
    eligible: &[bool],
    instrument_count: usize,
    date_idx: usize,
) -> Option<StandardizedWindow> {
    if date_idx + 1 < WINDOW {
        return None;
    }
    let start = date_idx + 1 - WINDOW;
    let mut codes = Vec::new();
    let mut vectors = Vec::new();

    for code_idx in 0..instrument_count {
        if !eligible[code_idx] {
            continue;
        }
        let mut raw = [0.0; WINDOW];
        let mut sum = 0.0;
        let mut valid = true;
        for (pos, day_idx) in (start..=date_idx).enumerate() {
            let Some(value) = clean(returns[day_idx * instrument_count + code_idx]) else {
                valid = false;
                break;
            };
            raw[pos] = value;
            sum += value;
        }
        if !valid {
            continue;
        }
        let mean = sum / WINDOW as f64;
        let mut sum_sq = 0.0;
        for value in &mut raw {
            *value -= mean;
            sum_sq += *value * *value;
        }
        if sum_sq <= f64::EPSILON {
            continue;
        }
        let norm = sum_sq.sqrt();
        for value in &mut raw {
            *value /= norm;
        }
        codes.push(code_idx);
        vectors.push(raw);
    }

    Some(StandardizedWindow { codes, vectors })
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    left: usize,
    right: usize,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct HeapEdge {
    edge: Edge,
}

impl Eq for HeapEdge {}

impl PartialEq for HeapEdge {
    fn eq(&self, other: &Self) -> bool {
        self.edge.weight.to_bits() == other.edge.weight.to_bits()
            && self.edge.left == other.edge.left
            && self.edge.right == other.edge.right
    }
}

impl Ord for HeapEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .edge
            .weight
            .total_cmp(&self.edge.weight)
            .then_with(|| other.edge.left.cmp(&self.edge.left))
            .then_with(|| other.edge.right.cmp(&self.edge.right))
    }
}

impl PartialOrd for HeapEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn top_abs_correlation_edges(vectors: &[[f64; WINDOW]], fraction: f64) -> Vec<Edge> {
    let node_count = vectors.len();
    if node_count < 2 || fraction <= 0.0 {
        return Vec::new();
    }
    let pair_count = node_count * (node_count - 1) / 2;
    let keep_count = ((pair_count as f64) * fraction).ceil() as usize;
    if keep_count == 0 {
        return Vec::new();
    }

    let mut heap = BinaryHeap::<HeapEdge>::with_capacity(keep_count + 1);
    for left in 0..node_count - 1 {
        for right in left + 1..node_count {
            let weight = dot(&vectors[left], &vectors[right]).clamp(-1.0, 1.0).abs();
            if !weight.is_finite() || weight <= f64::EPSILON {
                continue;
            }
            let candidate = HeapEdge {
                edge: Edge {
                    left,
                    right,
                    weight,
                },
            };
            if heap.len() < keep_count {
                heap.push(candidate);
            } else if heap
                .peek()
                .map(|current_min| candidate.edge.weight > current_min.edge.weight)
                .unwrap_or(false)
            {
                heap.pop();
                heap.push(candidate);
            }
        }
    }

    heap.into_iter().map(|item| item.edge).collect()
}

fn dot(left: &[f64; WINDOW], right: &[f64; WINDOW]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn adjacency_list(node_count: usize, edges: &[Edge]) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for edge in edges {
        if edge.weight <= f64::EPSILON || !edge.weight.is_finite() {
            continue;
        }
        adjacency[edge.left].push((edge.right, edge.weight));
        adjacency[edge.right].push((edge.left, edge.weight));
    }
    adjacency
}

fn landmark_indices(node_count: usize, max_landmarks: usize) -> Vec<usize> {
    if node_count == 0 || max_landmarks == 0 {
        return Vec::new();
    }
    if node_count <= max_landmarks {
        return (0..node_count).collect();
    }
    let mut output = Vec::with_capacity(max_landmarks);
    let last = node_count - 1;
    let denom = max_landmarks - 1;
    for idx in 0..max_landmarks {
        let landmark = (idx * last + denom / 2) / denom;
        if output.last().copied() != Some(landmark) {
            output.push(landmark);
        }
    }
    output
}

fn landmark_harmonic_centrality(
    node_count: usize,
    adjacency: &[Vec<(usize, f64)>],
    max_landmarks: usize,
) -> Vec<Option<f64>> {
    let landmarks = landmark_indices(node_count, max_landmarks);
    if node_count < 2 || landmarks.len() < 2 {
        return vec![None; node_count];
    }
    let mut sums = vec![0.0; node_count];
    let mut denominators = vec![landmarks.len(); node_count];
    for &source in &landmarks {
        denominators[source] = denominators[source].saturating_sub(1);
        let distances = dijkstra_distances(source, adjacency);
        for target in 0..node_count {
            if target == source {
                continue;
            }
            let distance = distances[target];
            if distance.is_finite() && distance > DIST_EPS {
                sums[target] += 1.0 / distance;
            }
        }
    }
    sums.into_iter()
        .zip(denominators)
        .map(|(sum, denominator)| {
            if denominator > 0 {
                let value = sum / denominator as f64;
                value.is_finite().then_some(value)
            } else {
                None
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct DijkstraState {
    node: usize,
    distance: f64,
}

impl Eq for DijkstraState {}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra_distances(source: usize, adjacency: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let mut distances = vec![f64::INFINITY; adjacency.len()];
    if source >= adjacency.len() {
        return distances;
    }
    let mut heap = BinaryHeap::new();
    distances[source] = 0.0;
    heap.push(DijkstraState {
        node: source,
        distance: 0.0,
    });

    while let Some(state) = heap.pop() {
        if state.distance > distances[state.node] {
            continue;
        }
        for &(next, weight) in &adjacency[state.node] {
            if weight <= f64::EPSILON {
                continue;
            }
            let edge_distance = 1.0 / weight;
            let candidate = state.distance + edge_distance;
            if candidate < distances[next] {
                distances[next] = candidate;
                heap.push(DijkstraState {
                    node: next,
                    distance: candidate,
                });
            }
        }
    }

    distances
}

fn constraint_centrality(node_count: usize, adjacency: &[Vec<(usize, f64)>]) -> Vec<Option<f64>> {
    let mut normalized = vec![Vec::<(usize, f64)>::new(); node_count];
    for node in 0..node_count {
        let row_sum = adjacency[node]
            .iter()
            .map(|(_, weight)| *weight)
            .filter(|weight| weight.is_finite() && *weight > 0.0)
            .sum::<f64>();
        if row_sum <= f64::EPSILON {
            continue;
        }
        normalized[node] = adjacency[node]
            .iter()
            .filter_map(|(next, weight)| {
                if weight.is_finite() && *weight > 0.0 {
                    Some((*next, *weight / row_sum))
                } else {
                    None
                }
            })
            .collect();
    }

    let mut output = vec![None; node_count];
    let mut scratch = vec![0.0; node_count];
    let mut touched = Vec::<usize>::new();
    let mut seen = vec![0usize; node_count];

    for node in 0..node_count {
        if normalized[node].is_empty() {
            continue;
        }
        let stamp = node + 1;
        touched.clear();
        for &(neighbor, direct) in &normalized[node] {
            add_scratch(
                neighbor,
                direct,
                stamp,
                &mut scratch,
                &mut touched,
                &mut seen,
            );
            for &(two_hop, neighbor_share) in &normalized[neighbor] {
                let value = direct * neighbor_share;
                add_scratch(two_hop, value, stamp, &mut scratch, &mut touched, &mut seen);
            }
        }
        let mut total = 0.0;
        for &idx in &touched {
            if idx == node {
                continue;
            }
            total += scratch[idx] * scratch[idx];
            scratch[idx] = 0.0;
            seen[idx] = 0;
        }
        if total.is_finite() {
            output[node] = Some(total);
        }
    }

    output
}

fn add_scratch(
    idx: usize,
    value: f64,
    stamp: usize,
    scratch: &mut [f64],
    touched: &mut Vec<usize>,
    seen: &mut [usize],
) {
    if idx >= scratch.len() || !value.is_finite() {
        return;
    }
    if seen[idx] != stamp {
        seen[idx] = stamp;
        scratch[idx] = 0.0;
        touched.push(idx);
    }
    scratch[idx] += value;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual={actual}, expected={expected}"
        );
    }

    fn unit_vector(entries: &[(usize, f64)]) -> [f64; WINDOW] {
        let mut values = [0.0; WINDOW];
        for &(idx, value) in entries {
            values[idx] = value;
        }
        let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
        for value in &mut values {
            *value /= norm;
        }
        values
    }

    #[test]
    fn rtn_corr_net_log_return_rejects_invalid_prices() {
        assert_close(log_return(Some(11.0), Some(10.0)), (1.1f64).ln());
        assert_eq!(log_return(Some(0.0), Some(10.0)), None);
        assert_eq!(log_return(Some(11.0), Some(0.0)), None);
    }

    #[test]
    fn rtn_corr_net_strict_window_requires_all_twenty_days_and_excludes_bj() {
        let instrument_count = 3;
        let eligible = vec![true, true, false];
        let mut returns = Vec::new();
        for day in 0..WINDOW {
            returns.push(Some(day as f64));
            returns.push(if day == 5 { None } else { Some(day as f64) });
            returns.push(Some(day as f64));
        }
        let window = strict_standardized_window(&returns, &eligible, instrument_count, WINDOW - 1)
            .expect("window");
        assert_eq!(window.codes, vec![0]);
    }

    #[test]
    fn rtn_corr_net_top_abs_edges_match_full_sort_on_small_sample() {
        let vectors = vec![
            unit_vector(&[(0, 1.0)]),
            unit_vector(&[(0, 0.8), (1, 0.6)]),
            unit_vector(&[(0, -0.3), (1, 0.4), (2, 0.8660254037844386)]),
            unit_vector(&[(1, 0.2), (2, -0.5), (3, 0.8426149773176358)]),
            unit_vector(&[(0, 0.1), (3, 0.6), (4, 0.7937253933193772)]),
        ];
        let mut expected = Vec::new();
        for left in 0..vectors.len() - 1 {
            for right in left + 1..vectors.len() {
                expected.push(Edge {
                    left,
                    right,
                    weight: dot(&vectors[left], &vectors[right]).abs(),
                });
            }
        }
        expected.sort_by(|left, right| right.weight.total_cmp(&left.weight));
        let keep = ((expected.len() as f64) * 0.3).ceil() as usize;
        expected.truncate(keep);
        expected.sort_by_key(|edge| (edge.left, edge.right));

        let mut actual = top_abs_correlation_edges(&vectors, 0.3);
        actual.sort_by_key(|edge| (edge.left, edge.right));

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!((actual.left, actual.right), (expected.left, expected.right));
            assert!((actual.weight - expected.weight).abs() < 1e-10);
        }
    }

    #[test]
    fn rtn_corr_net_landmarks_are_deterministic_and_cover_small_graphs() {
        assert_eq!(landmark_indices(0, 32), Vec::<usize>::new());
        assert_eq!(landmark_indices(4, 32), vec![0, 1, 2, 3]);
        assert_eq!(landmark_indices(5, 3), vec![0, 2, 4]);
    }

    #[test]
    fn rtn_corr_net_harmonic_uses_inverse_weight_shortest_paths() {
        let edges = vec![
            Edge {
                left: 0,
                right: 1,
                weight: 0.5,
            },
            Edge {
                left: 1,
                right: 2,
                weight: 1.0,
            },
        ];
        let adjacency = adjacency_list(3, &edges);
        let harmonic = landmark_harmonic_centrality(3, &adjacency, 3);

        assert_close(harmonic[0], (0.5 + 1.0 / 3.0) / 2.0);
        assert_close(harmonic[1], (0.5 + 1.0) / 2.0);
        assert_close(harmonic[2], (1.0 / 3.0 + 1.0) / 2.0);
    }

    #[test]
    fn rtn_corr_net_constraint_matches_manual_triangle() {
        let edges = vec![
            Edge {
                left: 0,
                right: 1,
                weight: 1.0,
            },
            Edge {
                left: 1,
                right: 2,
                weight: 1.0,
            },
        ];
        let adjacency = adjacency_list(3, &edges);
        let constraint = constraint_centrality(3, &adjacency);

        assert_close(constraint[0], 1.25);
        assert_close(constraint[1], 0.5);
        assert_close(constraint[2], 1.25);
    }

    #[test]
    fn rtn_corr_net_spec_has_cjzq_and_cs_network_tags() {
        let spec = StockDailyRtnCorrNetComposite.spec();
        assert_eq!(spec.id, "rtn_corr_net_composite");
        assert_eq!(spec.name, "rtn_corr_net_composite");
        assert!(spec.tags.iter().any(|tag| tag == "CJZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "cs_network"));
        assert!(spec.tags.iter().any(|tag| tag == "harmonic"));
        assert!(spec.tags.iter().any(|tag| tag == "constraint"));
        assert_eq!(spec.lookback.trading_days, WINDOW - 1);
    }

    #[test]
    fn rtn_corr_net_source_has_no_inner_parallelism_keywords() {
        let source = include_str!("rtn_corr_net_composite.rs");
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

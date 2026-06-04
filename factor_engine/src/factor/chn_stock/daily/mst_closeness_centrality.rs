use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const DIST_EPS: f64 = 1e-12;

pub struct StockDailyMstClosenessCentrality;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMstClosenessCentrality)
}

impl Factor for StockDailyMstClosenessCentrality {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "mst_closeness_centrality".to_string(),
            aliases: vec![
                "MSTClosenessCentrality".to_string(),
                "HCZQMSTCloseness".to_string(),
            ],
            name: "mst_closeness_centrality".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "HCZQ MST stock return-correlation network closeness centrality factor. It uses a strict 20-day daily log-return window, excludes BJ stocks, builds a minimum spanning tree from signed Pearson-correlation distances d=sqrt(2*(1-rho)), and outputs weighted tree closeness centrality without internal Rayon parallelism.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
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
        let values = mst_closeness_values(&panel, returns.values(), &eligible);
        let factor = panel.column_from_values(values)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "HCZQ",
        "cs_network",
        "mst",
        "correlation",
        "centrality",
        "closeness",
        "return",
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

fn mst_closeness_values(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for date_idx in WINDOW - 1..panel.dates().len() {
        let Some(window) =
            strict_standardized_window(returns, eligible, instrument_count, date_idx)
        else {
            continue;
        };
        if window.codes.len() < 2 {
            continue;
        }
        let Some(edges) = dense_prim_mst(&window.vectors) else {
            continue;
        };
        let closeness = closeness_centrality(window.codes.len(), &edges);
        let offset = date_idx * instrument_count;
        for (local_idx, code_idx) in window.codes.iter().enumerate() {
            output[offset + code_idx] = closeness[local_idx];
        }
    }

    output
}

struct StandardizedWindow {
    codes: Vec<usize>,
    vectors: Vec<Vec<f64>>,
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
        let mut centered = Vec::with_capacity(WINDOW);
        let mut sum_sq = 0.0;
        for value in raw {
            let centered_value = value - mean;
            centered.push(centered_value);
            sum_sq += centered_value * centered_value;
        }
        if sum_sq <= f64::EPSILON {
            continue;
        }
        let norm = sum_sq.sqrt();
        for value in &mut centered {
            *value /= norm;
        }
        codes.push(code_idx);
        vectors.push(centered);
    }

    Some(StandardizedWindow { codes, vectors })
}

#[derive(Clone, Copy, Debug)]
struct TreeEdge {
    left: usize,
    right: usize,
    weight: f64,
}

fn dense_prim_mst(vectors: &[Vec<f64>]) -> Option<Vec<TreeEdge>> {
    let node_count = vectors.len();
    if node_count < 2 {
        return None;
    }

    let mut in_tree = vec![false; node_count];
    let mut min_distance = vec![f64::INFINITY; node_count];
    let mut parent = vec![usize::MAX; node_count];
    min_distance[0] = 0.0;
    let mut edges = Vec::with_capacity(node_count - 1);

    for _ in 0..node_count {
        let mut best_idx = None;
        let mut best_distance = f64::INFINITY;
        for idx in 0..node_count {
            if !in_tree[idx] && min_distance[idx] < best_distance {
                best_distance = min_distance[idx];
                best_idx = Some(idx);
            }
        }
        let current = best_idx?;
        if !best_distance.is_finite() {
            return None;
        }
        in_tree[current] = true;
        if parent[current] != usize::MAX {
            edges.push(TreeEdge {
                left: parent[current],
                right: current,
                weight: best_distance,
            });
        }

        for candidate in 0..node_count {
            if in_tree[candidate] || candidate == current {
                continue;
            }
            let distance = correlation_distance(&vectors[current], &vectors[candidate]);
            if distance.is_finite() && distance < min_distance[candidate] {
                min_distance[candidate] = distance;
                parent[candidate] = current;
            }
        }
    }

    (edges.len() == node_count - 1).then_some(edges)
}

fn correlation_distance(left: &[f64], right: &[f64]) -> f64 {
    let rho = dot(left, right).clamp(-1.0, 1.0);
    (2.0 * (1.0 - rho)).max(0.0).sqrt()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[allow(dead_code)]
fn degree_centrality(node_count: usize, edges: &[TreeEdge]) -> Vec<Option<f64>> {
    if node_count < 2 {
        return vec![None; node_count];
    }
    let mut degree = vec![0usize; node_count];
    for edge in edges {
        if edge.left < node_count && edge.right < node_count {
            degree[edge.left] += 1;
            degree[edge.right] += 1;
        }
    }
    let denom = (node_count - 1) as f64;
    degree
        .into_iter()
        .map(|value| Some(value as f64 / denom))
        .collect()
}

fn closeness_centrality(node_count: usize, edges: &[TreeEdge]) -> Vec<Option<f64>> {
    if node_count < 2 || edges.len() != node_count - 1 {
        return vec![None; node_count];
    }
    let Some(tree) = TreeWork::new(node_count, edges) else {
        return vec![None; node_count];
    };
    let distance_sums = tree.distance_sums();
    distance_sums
        .into_iter()
        .map(|sum| {
            if sum > DIST_EPS && sum.is_finite() {
                Some((node_count - 1) as f64 / sum)
            } else {
                None
            }
        })
        .collect()
}

#[allow(dead_code)]
fn betweenness_centrality(node_count: usize, edges: &[TreeEdge]) -> Vec<Option<f64>> {
    if node_count < 2 || edges.len() != node_count - 1 {
        return vec![None; node_count];
    }
    let Some(tree) = TreeWork::new(node_count, edges) else {
        return vec![None; node_count];
    };
    let mut output = vec![Some(0.0); node_count];
    for node in 0..node_count {
        let mut prefix = 0usize;
        let mut total = 0usize;
        for (neighbor, _) in &tree.adjacency[node] {
            let part = if tree.parent[*neighbor] == node {
                tree.subtree_size[*neighbor]
            } else {
                node_count - tree.subtree_size[node]
            };
            total += prefix * part;
            prefix += part;
        }
        output[node] = Some(total as f64);
    }
    output
}

struct TreeWork {
    adjacency: Vec<Vec<(usize, f64)>>,
    parent: Vec<usize>,
    parent_weight: Vec<f64>,
    order: Vec<usize>,
    subtree_size: Vec<usize>,
    down_sum: Vec<f64>,
}

impl TreeWork {
    fn new(node_count: usize, edges: &[TreeEdge]) -> Option<Self> {
        let mut adjacency = vec![Vec::<(usize, f64)>::new(); node_count];
        for edge in edges {
            if edge.left >= node_count
                || edge.right >= node_count
                || !edge.weight.is_finite()
                || edge.weight < 0.0
            {
                return None;
            }
            adjacency[edge.left].push((edge.right, edge.weight));
            adjacency[edge.right].push((edge.left, edge.weight));
        }

        let mut parent = vec![usize::MAX; node_count];
        let mut parent_weight = vec![0.0; node_count];
        let mut order = Vec::with_capacity(node_count);
        let mut stack = vec![0usize];
        parent[0] = 0;
        while let Some(node) = stack.pop() {
            order.push(node);
            for (neighbor, weight) in &adjacency[node] {
                if parent[*neighbor] != usize::MAX {
                    continue;
                }
                parent[*neighbor] = node;
                parent_weight[*neighbor] = *weight;
                stack.push(*neighbor);
            }
        }
        if order.len() != node_count {
            return None;
        }

        let mut subtree_size = vec![1usize; node_count];
        let mut down_sum = vec![0.0; node_count];
        for node in order.iter().rev().copied() {
            for (neighbor, weight) in &adjacency[node] {
                if parent[*neighbor] == node {
                    subtree_size[node] += subtree_size[*neighbor];
                    down_sum[node] += down_sum[*neighbor] + subtree_size[*neighbor] as f64 * weight;
                }
            }
        }

        Some(Self {
            adjacency,
            parent,
            parent_weight,
            order,
            subtree_size,
            down_sum,
        })
    }

    fn distance_sums(&self) -> Vec<f64> {
        let node_count = self.adjacency.len();
        let mut sums = vec![0.0; node_count];
        sums[0] = self.down_sum[0];
        for node in self.order.iter().copied().skip(1) {
            let parent = self.parent[node];
            let weight = self.parent_weight[node];
            sums[node] =
                sums[parent] + (node_count as f64 - 2.0 * self.subtree_size[node] as f64) * weight;
        }
        sums
    }
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

    #[test]
    fn mst_log_return_rejects_invalid_prices() {
        assert!(log_return(Some(1.0), Some(0.0)).is_none());
        assert!(log_return(Some(-1.0), Some(1.0)).is_none());
        assert_close(log_return(Some(2.0), Some(1.0)), 2.0f64.ln());
    }

    #[test]
    fn mst_strict_window_requires_all_twenty_days_and_excludes_bj() {
        let instrument_count = 3;
        let eligible = vec![true, true, false];
        let mut returns = Vec::new();
        for day in 0..WINDOW {
            returns.push(Some(day as f64));
            returns.push(if day == 7 { None } else { Some(day as f64) });
            returns.push(Some(day as f64));
        }
        let window = strict_standardized_window(&returns, &eligible, instrument_count, WINDOW - 1)
            .expect("window");
        assert_eq!(window.codes, vec![0]);
    }

    #[test]
    fn mst_signed_correlation_distance_keeps_negative_farther() {
        let left = standardized(1.0);
        let same = standardized(2.0);
        let opposite = standardized(-1.0);
        let same_distance = correlation_distance(&left, &same);
        let opposite_distance = correlation_distance(&left, &opposite);
        assert!((same_distance - 0.0).abs() < 1e-6);
        assert!((opposite_distance - 2.0).abs() < 1e-6);
        assert!(opposite_distance > same_distance);
    }

    #[test]
    fn mst_dense_prim_builds_expected_tree() {
        let vectors = vec![
            standardized(1.0),
            standardized(2.0),
            standardized(-1.0),
            standardized(-2.0),
        ];
        let edges = dense_prim_mst(&vectors).expect("mst");
        assert_eq!(edges.len(), 3);
        assert_eq!(edges.iter().filter(|edge| edge.weight < 1e-6).count(), 2);
        assert!(edges.iter().any(|edge| {
            ((edge.left == 0 && edge.right == 2) || (edge.left == 2 && edge.right == 0))
                && (edge.weight - 2.0).abs() < 1e-10
        }));
    }

    #[test]
    fn mst_centrality_helpers_match_manual_tree() {
        let edges = vec![
            TreeEdge {
                left: 0,
                right: 1,
                weight: 1.0,
            },
            TreeEdge {
                left: 1,
                right: 2,
                weight: 1.0,
            },
            TreeEdge {
                left: 1,
                right: 3,
                weight: 2.0,
            },
        ];
        let degree = degree_centrality(4, &edges);
        assert_close(degree[1], 1.0);
        assert_close(degree[0], 1.0 / 3.0);

        let closeness = closeness_centrality(4, &edges);
        assert_close(closeness[1], 3.0 / 4.0);
        assert_close(closeness[0], 3.0 / 6.0);
        assert!(closeness[1].expect("center") > closeness[0].expect("leaf"));

        let betweenness = betweenness_centrality(4, &edges);
        assert_close(betweenness[1], 3.0);
        assert_close(betweenness[0], 0.0);
    }

    #[test]
    fn mst_compute_outputs_only_closeness_centrality() {
        let instrument_count = 3;
        let eligible = vec![true, true, true];
        let mut returns = Vec::new();
        for day in 0..WINDOW {
            let x = day as f64;
            returns.push(Some(x));
            returns.push(Some(1.1 * x));
            returns.push(Some(-x));
        }
        let panel = DailyPanel::from_index(
            (0..WINDOW).map(|idx| 20260101 + idx as i32).collect(),
            vec![
                "000001.SZ".to_string(),
                "000002.SZ".to_string(),
                "000003.SZ".to_string(),
            ],
            &[20260101 + WINDOW as i32 - 1],
            vec![true; WINDOW * instrument_count],
        )
        .expect("panel");
        let values = mst_closeness_values(&panel, &returns, &eligible);
        assert!(values[(WINDOW - 1) * instrument_count].is_some());
        assert!(values[(WINDOW - 1) * instrument_count + 1].is_some());
    }

    #[test]
    fn mst_spec_has_hczq_and_cs_network_tags() {
        let spec = StockDailyMstClosenessCentrality.spec();
        assert_eq!(spec.id, "mst_closeness_centrality");
        assert_eq!(spec.name, "mst_closeness_centrality");
        assert!(spec.tags.iter().any(|tag| tag == "HCZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "cs_network"));
        assert!(spec.tags.iter().any(|tag| tag == "mst"));
        assert!(spec.tags.iter().any(|tag| tag == "closeness"));
        assert_eq!(spec.lookback.trading_days, WINDOW - 1);
    }

    #[test]
    fn mst_source_has_no_inner_parallelism_keywords() {
        let source = include_str!("mst_closeness_centrality.rs");
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

    fn standardized(scale: f64) -> Vec<f64> {
        let raw = (0..WINDOW)
            .map(|idx| scale * idx as f64)
            .collect::<Vec<_>>();
        let mean = raw.iter().sum::<f64>() / WINDOW as f64;
        let mut centered = raw
            .into_iter()
            .map(|value| value - mean)
            .collect::<Vec<_>>();
        let norm = centered
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        for value in &mut centered {
            *value /= norm;
        }
        centered
    }
}

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
const EIGEN_KEEP_RATIO: f64 = 0.95;
const EDGE_THRESHOLD: f64 = 0.1;
const MAX_POWER_ITER: usize = 100;
const POWER_TOL: f64 = 1e-8;
const JACOBI_TOL: f64 = 1e-12;
const JACOBI_MAX_ITER: usize = 20_000;

pub struct StockDailyDenoisedEigenCentrality;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyDenoisedEigenCentrality)
}

impl Factor for StockDailyDenoisedEigenCentrality {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "denoised_eigen_centrality".to_string(),
            aliases: vec![
                "DenoisedEigenCentrality".to_string(),
                "DFZQEigenCentrality".to_string(),
            ],
            name: "denoised_eigen_centrality".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DFZQ denoised cross-sectional return-correlation network centrality factor. It uses a strict 20-day log-return window, denoises the return correlation network by retaining eigen components explaining 95% variance, keeps edges with abs(denoised corr) > 0.1, and outputs eigenvector centrality without internal Rayon parallelism.".to_string(),
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
        let values = denoised_eigen_centrality_values(&panel, returns.values(), &eligible);
        let factor = panel.column_from_values(values)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "DFZQ",
        "cs_network",
        "correlation",
        "centrality",
        "denoised",
        "eigenvector",
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

fn denoised_eigen_centrality_values(
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
        let embeddings = denoised_embeddings(&window.vectors);
        let edges = threshold_edges(&embeddings, EDGE_THRESHOLD);
        if edges.is_empty() {
            continue;
        }
        let centrality =
            eigenvector_centrality(window.codes.len(), &edges, MAX_POWER_ITER, POWER_TOL);
        let offset = date_idx * instrument_count;
        for (local_idx, code_idx) in window.codes.iter().enumerate() {
            output[offset + code_idx] = centrality[local_idx];
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

#[derive(Clone)]
struct DenoisedEmbedding {
    coords: Vec<f64>,
    norm_sq: f64,
}

fn denoised_embeddings(vectors: &[Vec<f64>]) -> Vec<Option<DenoisedEmbedding>> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let gram = gram_matrix(vectors, WINDOW);
    let eigens = jacobi_eigen_symmetric(gram, WINDOW);
    let retained = retained_eigenvectors(&eigens, EIGEN_KEEP_RATIO);
    if retained.is_empty() {
        return vec![None; vectors.len()];
    }

    vectors
        .iter()
        .map(|vector| {
            let coords = retained
                .iter()
                .map(|eigen| dot(vector, &eigen.vector))
                .collect::<Vec<_>>();
            let norm_sq = coords.iter().map(|value| value * value).sum::<f64>();
            if norm_sq > f64::EPSILON && norm_sq.is_finite() {
                Some(DenoisedEmbedding { coords, norm_sq })
            } else {
                None
            }
        })
        .collect()
}

fn gram_matrix(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut gram = vec![0.0; dim * dim];
    for vector in vectors {
        for left in 0..dim {
            let left_value = vector[left];
            for right in 0..=left {
                gram[left * dim + right] += left_value * vector[right];
            }
        }
    }
    for left in 0..dim {
        for right in 0..left {
            gram[right * dim + left] = gram[left * dim + right];
        }
    }
    gram
}

#[derive(Clone)]
struct EigenPair {
    value: f64,
    vector: Vec<f64>,
}

fn jacobi_eigen_symmetric(mut matrix: Vec<f64>, dim: usize) -> Vec<EigenPair> {
    let mut vectors = vec![0.0; dim * dim];
    for idx in 0..dim {
        vectors[idx * dim + idx] = 1.0;
    }

    for _ in 0..JACOBI_MAX_ITER {
        let Some((p, q, offdiag)) = largest_offdiag(&matrix, dim) else {
            break;
        };
        if offdiag < JACOBI_TOL {
            break;
        }
        let app = matrix[p * dim + p];
        let aqq = matrix[q * dim + q];
        let apq = matrix[p * dim + q];
        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = theta.cos();
        let s = theta.sin();

        for k in 0..dim {
            if k == p || k == q {
                continue;
            }
            let akp = matrix[k * dim + p];
            let akq = matrix[k * dim + q];
            let new_kp = c * akp - s * akq;
            let new_kq = s * akp + c * akq;
            matrix[k * dim + p] = new_kp;
            matrix[p * dim + k] = new_kp;
            matrix[k * dim + q] = new_kq;
            matrix[q * dim + k] = new_kq;
        }

        matrix[p * dim + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q * dim + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p * dim + q] = 0.0;
        matrix[q * dim + p] = 0.0;

        for k in 0..dim {
            let vip = vectors[k * dim + p];
            let viq = vectors[k * dim + q];
            vectors[k * dim + p] = c * vip - s * viq;
            vectors[k * dim + q] = s * vip + c * viq;
        }
    }

    let mut output = (0..dim)
        .map(|idx| EigenPair {
            value: matrix[idx * dim + idx],
            vector: (0..dim).map(|row| vectors[row * dim + idx]).collect(),
        })
        .collect::<Vec<_>>();
    output.sort_by(|left, right| right.value.total_cmp(&left.value));
    output
}

fn largest_offdiag(matrix: &[f64], dim: usize) -> Option<(usize, usize, f64)> {
    let mut best = None;
    let mut best_abs = 0.0;
    for left in 0..dim {
        for right in left + 1..dim {
            let value = matrix[left * dim + right].abs();
            if value > best_abs {
                best_abs = value;
                best = Some((left, right, value));
            }
        }
    }
    best
}

fn retained_eigenvectors(eigens: &[EigenPair], keep_ratio: f64) -> Vec<EigenPair> {
    let positive_total = eigens
        .iter()
        .filter(|eigen| eigen.value > f64::EPSILON)
        .map(|eigen| eigen.value)
        .sum::<f64>();
    if positive_total <= f64::EPSILON {
        return Vec::new();
    }

    let mut retained = Vec::new();
    let mut cumulative = 0.0;
    for eigen in eigens {
        if eigen.value <= f64::EPSILON {
            continue;
        }
        cumulative += eigen.value;
        retained.push(eigen.clone());
        if cumulative / positive_total >= keep_ratio {
            break;
        }
    }
    retained
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    left: usize,
    right: usize,
    weight: f64,
}

fn threshold_edges(embeddings: &[Option<DenoisedEmbedding>], threshold: f64) -> Vec<Edge> {
    let mut edges = Vec::new();
    if embeddings.len() < 2 {
        return edges;
    }

    for left in 0..embeddings.len() - 1 {
        let Some(left_embedding) = &embeddings[left] else {
            continue;
        };
        for right in left + 1..embeddings.len() {
            let Some(right_embedding) = &embeddings[right] else {
                continue;
            };
            let denominator = (left_embedding.norm_sq * right_embedding.norm_sq).sqrt();
            if denominator <= f64::EPSILON {
                continue;
            }
            let corr = dot(&left_embedding.coords, &right_embedding.coords) / denominator;
            let weight = corr.clamp(-1.0, 1.0).abs();
            if corr.is_finite() && weight > threshold {
                edges.push(Edge {
                    left,
                    right,
                    weight,
                });
            }
        }
    }

    edges
}

fn eigenvector_centrality(
    node_count: usize,
    edges: &[Edge],
    max_iter: usize,
    tolerance: f64,
) -> Vec<Option<f64>> {
    if node_count == 0 {
        return Vec::new();
    }
    if edges.is_empty() {
        return vec![None; node_count];
    }
    let initial = 1.0 / (node_count as f64).sqrt();
    let mut current = vec![initial; node_count];

    for _ in 0..max_iter {
        let mut next = vec![0.0; node_count];
        for edge in edges {
            next[edge.left] += edge.weight * current[edge.right];
            next[edge.right] += edge.weight * current[edge.left];
        }
        let norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm <= f64::EPSILON || !norm.is_finite() {
            return vec![None; node_count];
        }
        for value in &mut next {
            *value /= norm;
        }
        let diff = next
            .iter()
            .zip(&current)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        current = next;
        if diff < tolerance {
            break;
        }
    }

    current
        .into_iter()
        .map(|value| value.is_finite().then_some(value))
        .collect()
}

#[allow(dead_code)]
fn weighted_degree_centrality(node_count: usize, edges: &[Edge]) -> Vec<Option<f64>> {
    let mut output = vec![0.0; node_count];
    for edge in edges {
        output[edge.left] += edge.weight;
        output[edge.right] += edge.weight;
    }
    output.into_iter().map(Some).collect()
}

#[allow(dead_code)]
fn weighted_closeness_centrality(node_count: usize, edges: &[Edge]) -> Vec<Option<f64>> {
    let adjacency = adjacency_list(node_count, edges);
    (0..node_count)
        .map(|source| {
            let distances = dijkstra_distances(source, &adjacency);
            let mut sum = 0.0;
            let mut reachable = 0usize;
            for (idx, distance) in distances.iter().enumerate() {
                if idx == source {
                    continue;
                }
                if distance.is_finite() {
                    sum += distance;
                    reachable += 1;
                }
            }
            if reachable > 0 && sum > f64::EPSILON {
                Some(reachable as f64 / sum)
            } else {
                None
            }
        })
        .collect()
}

#[allow(dead_code)]
fn adjacency_list(node_count: usize, edges: &[Edge]) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for edge in edges {
        if edge.weight <= f64::EPSILON {
            continue;
        }
        let distance = 1.0 / edge.weight;
        adjacency[edge.left].push((edge.right, distance));
        adjacency[edge.right].push((edge.left, distance));
    }
    adjacency
}

#[allow(dead_code)]
fn dijkstra_distances(source: usize, adjacency: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let node_count = adjacency.len();
    let mut distances = vec![f64::INFINITY; node_count];
    let mut visited = vec![false; node_count];
    distances[source] = 0.0;

    for _ in 0..node_count {
        let mut best_idx = None;
        let mut best_distance = f64::INFINITY;
        for idx in 0..node_count {
            if !visited[idx] && distances[idx] < best_distance {
                best_distance = distances[idx];
                best_idx = Some(idx);
            }
        }
        let Some(current) = best_idx else {
            break;
        };
        visited[current] = true;
        for (next, distance) in &adjacency[current] {
            let candidate = distances[current] + distance;
            if candidate < distances[*next] {
                distances[*next] = candidate;
            }
        }
    }

    distances
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

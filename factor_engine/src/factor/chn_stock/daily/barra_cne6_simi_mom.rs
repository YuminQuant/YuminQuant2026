use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::neutralize::CNE6_PRIMARY_BARRA_COLUMNS;
use crate::operators::{cs_regression_residual, cs_zscore};

const VERSION: &str = "0.1.0";
const STYLE_DIM: usize = 9;
const STYLE_RADIUS: f64 = 3.0;
const MOMENTUM_BASE_LAG: usize = 252;
const MOMENTUM_CURRENT_LAG: usize = 21;
const REVERSAL_BASE_LAG: usize = 21;
const REVERSAL_CURRENT_LAG: usize = 0;

pub struct StockDailyBarraCne6SimiMom;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyBarraCne6SimiMom)
}

impl Factor for StockDailyBarraCne6SimiMom {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "barra_cne6_simi_mom".to_string(),
            aliases: vec![
                "Barra CNE6 Simi Mom".to_string(),
                "style_similarity_composite".to_string(),
            ],
            name: "barra_cne6_simi_mom".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "CJZQ CNE6 style-similarity cross-sectional network factor. It links non-BJ stocks whose latest 9-dimensional Barra CNE6 exposure Euclidean distance is <= 3, combines orthogonalized peer Ret(252,21), peer Ret(21,0), and own Ret(21,0), then neutralizes by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &CNE6_PRIMARY_BARRA_COLUMNS),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: MOMENTUM_BASE_LAG,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let style_columns = cne6_style_columns(&panel, data)?;
        let adj_close = adjusted_close(&panel, data)?;
        let self_momentum = adj_close.ts(|series| {
            lagged_price_return_series(series, MOMENTUM_CURRENT_LAG, MOMENTUM_BASE_LAG)
        })?;
        let self_reversal = adj_close.ts(|series| {
            lagged_price_return_series(series, REVERSAL_CURRENT_LAG, REVERSAL_BASE_LAG)
        })?;

        let (related_momentum, related_reversal) =
            style_similarity_peer_averages(&style_columns, &self_momentum, &self_reversal, &panel)?;
        let related_momentum_residual =
            related_momentum.cs_binary(&self_momentum, cs_regression_residual)?;

        let composite = average_three(
            &related_momentum_residual.cs(cs_zscore)?,
            &related_reversal.cs(cs_zscore)?,
            &self_reversal.cs(cs_zscore)?,
        )?;
        let factor = neutralize_size_sector(&composite, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "CJZQ",
        "cs_network",
        "style_similarity",
        "cne6",
        "barra",
        "momentum",
        "reversal",
        "neutralize",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn cne6_style_columns(panel: &DailyPanel, data: &DataPool) -> Result<Vec<PanelColumn>> {
    let barra = data.daily(DatasetId::StockBarraDaily)?;
    CNE6_PRIMARY_BARRA_COLUMNS
        .iter()
        .map(|column| panel.column_from_table(barra, column))
        .collect()
}

fn adjusted_close(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let pv = data.daily(DatasetId::StockDailyPv)?;
    let close = panel.column_from_table(pv, "close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    close.zip_binary(&adj_factor, |close, adj_factor| {
        let (Some(close), Some(adj_factor)) = (clean(close), clean(adj_factor)) else {
            return None;
        };
        let value = close * adj_factor;
        value.is_finite().then_some(value)
    })
}

fn lagged_price_return_series(
    values: &[Option<f64>],
    current_lag: usize,
    base_lag: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if base_lag <= current_lag {
        return output;
    }
    for idx in base_lag..values.len() {
        let current_idx = idx - current_lag;
        let base_idx = idx - base_lag;
        let (Some(current), Some(base)) = (clean(values[current_idx]), clean(values[base_idx]))
        else {
            continue;
        };
        if base.abs() <= f64::EPSILON {
            continue;
        }
        let value = current / base - 1.0;
        if value.is_finite() {
            output[idx] = Some(value);
        }
    }
    output
}

fn style_similarity_peer_averages(
    style_columns: &[PanelColumn],
    self_momentum: &PanelColumn,
    self_reversal: &PanelColumn,
    panel: &DailyPanel,
) -> Result<(PanelColumn, PanelColumn)> {
    let code_count = panel.instruments().len();
    let eligible = eligible_instruments(panel);
    let mut related_momentum = vec![None; panel.shape_len()];
    let mut related_reversal = vec![None; panel.shape_len()];

    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * code_count;
        let points = style_points_for_date(style_columns, offset, code_count, &eligible);
        let momentum = &self_momentum.values()[offset..offset + code_count];
        let reversal = &self_reversal.values()[offset..offset + code_count];
        let (day_momentum, day_reversal) =
            peer_averages_for_points(&points, momentum, reversal, code_count, STYLE_RADIUS);
        for code_idx in 0..code_count {
            related_momentum[offset + code_idx] = day_momentum[code_idx];
            related_reversal[offset + code_idx] = day_reversal[code_idx];
        }
    }

    Ok((
        panel.column_from_values(related_momentum)?,
        panel.column_from_values(related_reversal)?,
    ))
}

fn eligible_instruments(panel: &DailyPanel) -> Vec<bool> {
    panel
        .instruments()
        .iter()
        .map(|ts_code| !is_bj_stock(ts_code))
        .collect()
}

fn style_points_for_date(
    style_columns: &[PanelColumn],
    offset: usize,
    code_count: usize,
    eligible: &[bool],
) -> Vec<StylePoint> {
    let mut points = Vec::new();
    for code_idx in 0..code_count {
        if !eligible[code_idx] {
            continue;
        }
        if let Some(values) = style_vector_at(style_columns, offset + code_idx) {
            points.push(StylePoint {
                instrument_idx: code_idx,
                values,
            });
        }
    }
    points
}

fn style_vector_at(style_columns: &[PanelColumn], panel_idx: usize) -> Option<[f64; STYLE_DIM]> {
    if style_columns.len() != STYLE_DIM {
        return None;
    }
    let mut values = [0.0; STYLE_DIM];
    for dim in 0..STYLE_DIM {
        values[dim] = clean(style_columns[dim].values()[panel_idx])?;
    }
    Some(values)
}

#[derive(Clone, Copy, Debug)]
struct StylePoint {
    instrument_idx: usize,
    values: [f64; STYLE_DIM],
}

fn peer_averages_for_points(
    points: &[StylePoint],
    self_momentum: &[Option<f64>],
    self_reversal: &[Option<f64>],
    instrument_count: usize,
    radius: f64,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let mut momentum_sum = vec![0.0; instrument_count];
    let mut momentum_count = vec![0usize; instrument_count];
    let mut reversal_sum = vec![0.0; instrument_count];
    let mut reversal_count = vec![0usize; instrument_count];

    if points.len() >= 2 {
        let tree = KdTree::new(points);
        let radius_sq = radius * radius;
        for left_point_idx in 0..points.len() {
            let mut neighbors = Vec::new();
            tree.query_radius(left_point_idx, radius_sq, &mut neighbors);
            for right_point_idx in neighbors {
                if right_point_idx <= left_point_idx {
                    continue;
                }
                accumulate_peer_pair(
                    points[left_point_idx].instrument_idx,
                    points[right_point_idx].instrument_idx,
                    self_momentum,
                    &mut momentum_sum,
                    &mut momentum_count,
                );
                accumulate_peer_pair(
                    points[left_point_idx].instrument_idx,
                    points[right_point_idx].instrument_idx,
                    self_reversal,
                    &mut reversal_sum,
                    &mut reversal_count,
                );
            }
        }
    }

    (
        average_from_sum_count(momentum_sum, momentum_count),
        average_from_sum_count(reversal_sum, reversal_count),
    )
}

fn accumulate_peer_pair(
    left: usize,
    right: usize,
    values: &[Option<f64>],
    sums: &mut [f64],
    counts: &mut [usize],
) {
    if let Some(right_value) = clean(values[right]) {
        sums[left] += right_value;
        counts[left] += 1;
    }
    if let Some(left_value) = clean(values[left]) {
        sums[right] += left_value;
        counts[right] += 1;
    }
}

fn average_from_sum_count(sums: Vec<f64>, counts: Vec<usize>) -> Vec<Option<f64>> {
    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                return None;
            }
            let value = sum / count as f64;
            value.is_finite().then_some(value)
        })
        .collect()
}

fn average_three(
    left: &PanelColumn,
    middle: &PanelColumn,
    right: &PanelColumn,
) -> Result<PanelColumn> {
    left.zip_ternary(middle, right, |left, middle, right| {
        match (clean(left), clean(middle), clean(right)) {
            (Some(left), Some(middle), Some(right)) => {
                let value = (left + middle + right) / 3.0;
                value.is_finite().then_some(value)
            }
            _ => None,
        }
    })
}

fn distance_squared(left: &[f64; STYLE_DIM], right: &[f64; STYLE_DIM]) -> f64 {
    let mut sum = 0.0;
    for dim in 0..STYLE_DIM {
        let diff = left[dim] - right[dim];
        sum += diff * diff;
    }
    sum
}

#[derive(Debug)]
struct KdTree<'a> {
    points: &'a [StylePoint],
    nodes: Vec<KdNode>,
    root: Option<usize>,
}

#[derive(Debug)]
struct KdNode {
    point_idx: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

impl<'a> KdTree<'a> {
    fn new(points: &'a [StylePoint]) -> Self {
        let mut order = (0..points.len()).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(points.len());
        let root = build_kd_node(&mut order, 0, points, &mut nodes);
        Self {
            points,
            nodes,
            root,
        }
    }

    fn query_radius(&self, target_point_idx: usize, radius_sq: f64, output: &mut Vec<usize>) {
        let Some(root) = self.root else {
            return;
        };
        self.query_node(root, target_point_idx, radius_sq, output);
    }

    fn query_node(
        &self,
        node_idx: usize,
        target_point_idx: usize,
        radius_sq: f64,
        output: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];
        let target = &self.points[target_point_idx];
        let candidate = &self.points[node.point_idx];
        if node.point_idx != target_point_idx
            && distance_squared(&target.values, &candidate.values) <= radius_sq
        {
            output.push(node.point_idx);
        }

        let axis = node.axis;
        let diff = target.values[axis] - candidate.values[axis];
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(child) = near {
            self.query_node(child, target_point_idx, radius_sq, output);
        }
        if diff * diff <= radius_sq {
            if let Some(child) = far {
                self.query_node(child, target_point_idx, radius_sq, output);
            }
        }
    }
}

fn build_kd_node(
    order: &mut [usize],
    depth: usize,
    points: &[StylePoint],
    nodes: &mut Vec<KdNode>,
) -> Option<usize> {
    if order.is_empty() {
        return None;
    }
    let axis = depth % STYLE_DIM;
    let median = order.len() / 2;
    order.select_nth_unstable_by(median, |left, right| {
        points[*left].values[axis].total_cmp(&points[*right].values[axis])
    });

    let point_idx = order[median];
    let node_idx = nodes.len();
    nodes.push(KdNode {
        point_idx,
        axis,
        left: None,
        right: None,
    });

    let (left_order, right_with_median) = order.split_at_mut(median);
    let right_order = &mut right_with_median[1..];
    let left = build_kd_node(left_order, depth + 1, points, nodes);
    let right = build_kd_node(right_order, depth + 1, points, nodes);
    nodes[node_idx].left = left;
    nodes[node_idx].right = right;
    Some(node_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(instrument_idx: usize, first_dim: f64) -> StylePoint {
        let mut values = [0.0; STYLE_DIM];
        values[0] = first_dim;
        StylePoint {
            instrument_idx,
            values,
        }
    }

    fn brute_neighbors(points: &[StylePoint], idx: usize, radius: f64) -> Vec<usize> {
        let radius_sq = radius * radius;
        let mut output = points
            .iter()
            .enumerate()
            .filter_map(|(candidate_idx, candidate)| {
                if candidate_idx != idx
                    && distance_squared(&points[idx].values, &candidate.values) <= radius_sq
                {
                    Some(candidate_idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        output.sort_unstable();
        output
    }

    #[test]
    fn barra_cne6_simi_mom_lagged_return_uses_requested_lags() {
        let values = (1..=260)
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();
        let momentum = lagged_price_return_series(&values, 21, 252);
        let expected = values[252 - 21].unwrap() / values[0].unwrap() - 1.0;
        assert!((momentum[252].unwrap() - expected).abs() < 1e-12);

        let reversal = lagged_price_return_series(&values, 0, 21);
        let expected = values[21].unwrap() / values[0].unwrap() - 1.0;
        assert!((reversal[21].unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn barra_cne6_simi_mom_distance_uses_all_style_dimensions() {
        let mut left = [0.0; STYLE_DIM];
        let mut right = [0.0; STYLE_DIM];
        right[0] = 1.0;
        right[8] = 2.0;
        assert!((distance_squared(&left, &right) - 5.0).abs() < 1e-12);
        left[8] = 2.0;
        assert!((distance_squared(&left, &right) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn barra_cne6_simi_mom_kdtree_matches_bruteforce_radius_search() {
        let points = vec![
            point(0, 0.0),
            point(1, 1.0),
            point(2, 2.9),
            point(3, 3.1),
            point(4, 7.0),
        ];
        let tree = KdTree::new(&points);
        for idx in 0..points.len() {
            let mut actual = Vec::new();
            tree.query_radius(idx, 3.0 * 3.0, &mut actual);
            actual.sort_unstable();
            assert_eq!(actual, brute_neighbors(&points, idx, 3.0));
        }
    }

    #[test]
    fn barra_cne6_simi_mom_peer_averages_exclude_self_and_use_valid_peers() {
        let points = vec![point(0, 0.0), point(1, 1.0), point(2, 7.0)];
        let momentum = vec![Some(0.1), Some(0.2), Some(0.9)];
        let reversal = vec![Some(-0.1), None, Some(-0.9)];
        let (peer_momentum, peer_reversal) =
            peer_averages_for_points(&points, &momentum, &reversal, 3, 3.0);

        assert!((peer_momentum[0].unwrap() - 0.2).abs() < 1e-12);
        assert!((peer_momentum[1].unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(peer_momentum[2], None);
        assert_eq!(peer_reversal[0], None);
        assert!((peer_reversal[1].unwrap() + 0.1).abs() < 1e-12);
        assert_eq!(peer_reversal[2], None);
    }

    #[test]
    fn barra_cne6_simi_mom_spec_has_required_tags_and_lookback() {
        let spec = StockDailyBarraCne6SimiMom.spec();
        assert_eq!(spec.id, "barra_cne6_simi_mom");
        assert_eq!(spec.name, "barra_cne6_simi_mom");
        assert!(spec.tags.iter().any(|tag| tag == "CJZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "cs_network"));
        assert!(spec.tags.iter().any(|tag| tag == "cne6"));
        assert_eq!(spec.lookback.trading_days, MOMENTUM_BASE_LAG);
    }

    #[test]
    fn barra_cne6_simi_mom_source_has_no_inner_parallelism_keywords() {
        let source = include_str!("barra_cne6_simi_mom.rs");
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

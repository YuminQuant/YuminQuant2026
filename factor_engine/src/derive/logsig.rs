use std::collections::BTreeMap;

use crate::data::Table;
use crate::error::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct LogsigSignatureRow {
    pub trade_date: i32,
    pub ts_code: String,
    pub values: Vec<f32>,
}

pub fn derive_logsig_volume_signature_rows(
    trade_date: i32,
    tables: &[Table],
    lookback_days: usize,
    bar_size: usize,
    order: usize,
) -> Result<Vec<LogsigSignatureRow>> {
    let expected = lookback_days * (240 / bar_size);
    let mut by_symbol: BTreeMap<String, Vec<(i32, i32, f64)>> = BTreeMap::new();
    for table in tables {
        let dates = table.required_i32("trade_date")?;
        let symbols = table.required_utf8("ts_code")?;
        let bar_indices = table.required_i32("bar_index")?;
        let volumes = table.required_f64_cast("volume")?;
        for idx in 0..table.len {
            let Some(date) = dates[idx] else { continue };
            let Some(symbol) = symbols[idx].as_ref() else {
                continue;
            };
            let Some(bar_index) = bar_indices[idx] else {
                continue;
            };
            let Some(volume) = volumes[idx] else { continue };
            if volume.is_finite() {
                by_symbol.entry(symbol.clone()).or_default().push((
                    date,
                    bar_index,
                    volume.max(1.0).ln(),
                ));
            }
        }
    }

    let mut rows = Vec::new();
    for (symbol, mut values) in by_symbol {
        values.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        if values.len() != expected {
            continue;
        }
        let path = values
            .into_iter()
            .map(|(_, _, value)| value)
            .collect::<Vec<_>>();
        let signature = signature_of_lead_lag_path(&path, order)
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        rows.push(LogsigSignatureRow {
            trade_date,
            ts_code: symbol,
            values: signature,
        });
    }
    Ok(rows)
}

pub fn signature_width(order: usize) -> usize {
    (1..=order).map(|level| 2_usize.pow(level as u32)).sum()
}

pub fn lead_lag_path(values: &[f64]) -> Vec<[f64; 2]> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(values.len().saturating_mul(2).saturating_sub(1));
    output.push([values[0], values[0]]);
    for idx in 1..values.len() {
        output.push([values[idx], values[idx - 1]]);
        output.push([values[idx], values[idx]]);
    }
    output
}

pub fn signature_of_lead_lag_path(values: &[f64], order: usize) -> Vec<f64> {
    if order == 0 || values.is_empty() {
        return Vec::new();
    }
    let mut levels = empty_signature_levels(order);
    for window in values.windows(2) {
        let delta = window[1] - window[0];
        append_axis_segment(&mut levels, 0, delta, order);
        append_axis_segment(&mut levels, 1, delta, order);
    }
    levels.into_iter().skip(1).flatten().collect()
}

pub fn signature_of_path(path: &[[f64; 2]], order: usize) -> Vec<f64> {
    if order == 0 {
        return Vec::new();
    }
    let mut levels = empty_signature_levels(order);
    for segment in path.windows(2) {
        let dx = [segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]];
        if dx[1].abs() <= f64::EPSILON {
            append_axis_segment(&mut levels, 0, dx[0], order);
        } else if dx[0].abs() <= f64::EPSILON {
            append_axis_segment(&mut levels, 1, dx[1], order);
        } else {
            let increment = linear_increment_signature(dx, order);
            levels = chen_product(&levels, &increment, order);
        }
    }
    levels.into_iter().skip(1).flatten().collect()
}

fn empty_signature_levels(order: usize) -> Vec<Vec<f64>> {
    let mut levels = Vec::with_capacity(order + 1);
    levels.push(vec![1.0]);
    for level in 1..=order {
        levels.push(vec![0.0; 2_usize.pow(level as u32)]);
    }
    levels
}

fn append_axis_segment(levels: &mut [Vec<f64>], axis: usize, delta: f64, order: usize) {
    if delta.abs() <= f64::EPSILON {
        return;
    }
    let previous = levels.to_vec();
    let mut scaled_powers = vec![1.0; order + 1];
    for level in 1..=order {
        scaled_powers[level] = scaled_powers[level - 1] * delta / level as f64;
    }
    for level in 1..=order {
        let width = 2_usize.pow(level as u32);
        for word in 0..width {
            let suffix = repeated_axis_suffix_len(word, level, axis);
            let mut value = previous[level][word];
            for repeat in 1..=suffix {
                let prefix_word = word >> repeat;
                value += previous[level - repeat][prefix_word] * scaled_powers[repeat];
            }
            levels[level][word] = value;
        }
    }
}

fn repeated_axis_suffix_len(word: usize, level: usize, axis: usize) -> usize {
    let mut count = 0;
    for bit in 0..level {
        if ((word >> bit) & 1) == axis {
            count += 1;
        } else {
            break;
        }
    }
    count
}

fn linear_increment_signature(dx: [f64; 2], order: usize) -> Vec<Vec<f64>> {
    let mut levels = Vec::with_capacity(order + 1);
    levels.push(vec![1.0]);
    let mut factorial = 1.0;
    for level in 1..=order {
        factorial *= level as f64;
        let width = 2_usize.pow(level as u32);
        let mut values = Vec::with_capacity(width);
        for word in 0..width {
            let mut product = 1.0;
            for bit in (0..level).rev() {
                let coordinate = (word >> bit) & 1;
                product *= dx[coordinate];
            }
            values.push(product / factorial);
        }
        levels.push(values);
    }
    levels
}

fn chen_product(left: &[Vec<f64>], right: &[Vec<f64>], order: usize) -> Vec<Vec<f64>> {
    let mut output = Vec::with_capacity(order + 1);
    output.push(vec![1.0]);
    for level in 1..=order {
        let mut values = left[level].clone();
        for split in 1..level {
            let left_level = &left[split];
            let right_level = &right[level - split];
            let right_width = right_level.len();
            for (left_idx, left_value) in left_level.iter().enumerate() {
                for (right_idx, right_value) in right_level.iter().enumerate() {
                    values[left_idx * right_width + right_idx] += left_value * right_value;
                }
            }
        }
        for (idx, value) in right[level].iter().enumerate() {
            values[idx] += value;
        }
        output.push(values);
    }
    output
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::data::{ColumnData, Table};
    use crate::derive::logsig::{
        derive_logsig_volume_signature_rows, lead_lag_path, signature_of_path, signature_width,
    };

    #[test]
    fn lead_lag_path_advances_lead_then_lag() {
        assert_eq!(
            lead_lag_path(&[1.0, 2.0, 3.0]),
            vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [3.0, 2.0], [3.0, 3.0]]
        );
    }

    #[test]
    fn straight_line_signature_matches_increment_exponential() {
        let signature = signature_of_path(&[[0.0, 0.0], [2.0, 3.0]], 2);
        assert_eq!(signature_width(2), 6);
        assert!((signature[0] - 2.0).abs() < 1e-12);
        assert!((signature[1] - 3.0).abs() < 1e-12);
        assert!((signature[2] - 2.0).abs() < 1e-12);
        assert!((signature[3] - 3.0).abs() < 1e-12);
        assert!((signature[4] - 3.0).abs() < 1e-12);
        assert!((signature[5] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn lead_lag_fast_path_matches_generic_signature() {
        let values = [1.0, 2.0, 0.5, 3.0];
        let fast = super::signature_of_lead_lag_path(&values, 4);
        let generic = signature_of_path(&lead_lag_path(&values), 4);
        assert_eq!(fast.len(), generic.len());
        for (left, right) in fast.iter().zip(generic.iter()) {
            assert!((left - right).abs() < 1e-10);
        }
    }

    #[test]
    fn volume_rows_log_clip_volume_and_require_full_window() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260102), Some(20260102), Some(20260102)]),
            ),
            (
                "bar_index".to_string(),
                ColumnData::I32(vec![Some(0), Some(1), Some(0)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                ]),
            ),
            (
                "volume".to_string(),
                ColumnData::F64(vec![Some(0.0), Some(10.0), Some(5.0)]),
            ),
        ]))
        .expect("table");
        let rows =
            derive_logsig_volume_signature_rows(20260102, &[table], 1, 120, 2).expect("rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ts_code, "000001.SZ");
        assert_eq!(rows[0].values.len(), 6);
    }
}

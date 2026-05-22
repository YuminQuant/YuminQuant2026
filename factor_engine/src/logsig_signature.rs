use rayon::prelude::*;

use crate::error::{err, Result};

pub fn signature_width(order: usize) -> Result<usize> {
    if order == 0 {
        return Err(err("logsig signature order must be positive"));
    }
    let mut width = 0usize;
    for level in 1..=order {
        let level_width = 1usize
            .checked_shl(level as u32)
            .ok_or_else(|| err(format!("logsig signature order {order} is too large")))?;
        width = width
            .checked_add(level_width)
            .ok_or_else(|| err(format!("logsig signature order {order} is too large")))?;
    }
    Ok(width)
}

pub fn logsig_signature_batch_from_volume(
    volume: &[f64],
    rows: usize,
    cols: usize,
    order: usize,
) -> Result<Vec<f32>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if cols == 0 {
        return Err(err("logsig signature volume matrix must have at least one column"));
    }
    if volume.len() != rows.saturating_mul(cols) {
        return Err(err(format!(
            "logsig signature volume length {} does not match shape {rows}x{cols}",
            volume.len()
        )));
    }
    let width = signature_width(order)?;
    let level_offsets = level_offsets(order)?;
    let mut output = vec![0.0f32; rows * width];
    output
        .par_chunks_mut(width)
        .zip(volume.par_chunks(cols))
        .try_for_each(|(out, row)| compute_row(row, order, width, &level_offsets, out))?;
    Ok(output)
}

fn level_offsets(order: usize) -> Result<Vec<usize>> {
    let mut offsets = vec![0usize; order + 1];
    let mut running = 0usize;
    for level in 1..=order {
        offsets[level] = running;
        running = running
            .checked_add(1usize << level)
            .ok_or_else(|| err(format!("logsig signature order {order} is too large")))?;
    }
    Ok(offsets)
}

fn compute_row(
    volume: &[f64],
    order: usize,
    width: usize,
    level_offsets: &[usize],
    out: &mut [f32],
) -> Result<()> {
    let mut levels = vec![0.0f64; width];
    let mut previous = vec![0.0f64; width];
    let mut scaled = vec![0.0f64; order + 1];
    let mut previous_log = clipped_log(volume[0])?;
    for value in volume.iter().copied().skip(1) {
        let current_log = clipped_log(value)?;
        let delta = current_log - previous_log;
        previous_log = current_log;
        if delta.abs() <= 1e-15 {
            continue;
        }
        append_axis_segment(&mut levels, &mut previous, level_offsets, &mut scaled, 0, delta, order);
        append_axis_segment(&mut levels, &mut previous, level_offsets, &mut scaled, 1, delta, order);
    }
    for (dst, src) in out.iter_mut().zip(levels) {
        *dst = src as f32;
    }
    Ok(())
}

fn clipped_log(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(err("logsig signature volume contains non-finite value"));
    }
    Ok(value.max(1.0).ln())
}

fn append_axis_segment(
    levels: &mut [f64],
    previous: &mut [f64],
    level_offsets: &[usize],
    scaled: &mut [f64],
    axis: usize,
    delta: f64,
    order: usize,
) {
    previous.copy_from_slice(levels);
    scaled[0] = 1.0;
    for level in 1..=order {
        scaled[level] = scaled[level - 1] * delta / level as f64;
    }
    for level in 1..=order {
        let width = 1usize << level;
        let offset = level_offsets[level];
        for word in 0..width {
            let suffix = repeated_axis_suffix_len(word, level, axis);
            let mut value = previous[offset + word];
            for repeat in 1..=suffix {
                let prefix_value = if level == repeat {
                    1.0
                } else {
                    previous[level_offsets[level - repeat] + (word >> repeat)]
                };
                value += prefix_value * scaled[repeat];
            }
            levels[offset + word] = value;
        }
    }
}

fn repeated_axis_suffix_len(word: usize, level: usize, axis: usize) -> usize {
    let mut count = 0usize;
    for bit in 0..level {
        if ((word >> bit) & 1) == axis {
            count += 1;
        } else {
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_matches_order_ten() {
        assert_eq!(signature_width(10).unwrap(), 2046);
    }

    #[test]
    fn zero_volume_is_clipped_before_log() {
        let with_zero = logsig_signature_batch_from_volume(&[0.0, 10.0], 1, 2, 3).unwrap();
        let clipped = logsig_signature_batch_from_volume(&[1.0, 10.0], 1, 2, 3).unwrap();
        assert_close(&with_zero, &clipped, 1e-6);
    }

    #[test]
    fn two_point_path_matches_reference_order_three() {
        let actual = logsig_signature_batch_from_volume(&[1.0, 10.0], 1, 2, 3).unwrap();
        let reference = reference_signature(&[1.0f64.ln(), 10.0f64.ln()], 3);
        assert_close(&actual, &reference, 1e-6);
    }

    #[test]
    fn batch_preserves_row_order() {
        let actual = logsig_signature_batch_from_volume(&[1.0, 10.0, 1.0, 100.0], 2, 2, 1).unwrap();
        assert_eq!(actual.len(), 4);
        assert!(actual[0] < actual[2]);
        assert!(actual[1] < actual[3]);
    }

    #[test]
    fn rejects_non_finite_volume() {
        let error = logsig_signature_batch_from_volume(&[1.0, f64::NAN], 1, 2, 2).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    fn reference_signature(values: &[f64], order: usize) -> Vec<f32> {
        let mut state = vec![0.0f64; signature_width(order).unwrap()];
        let level_offsets = level_offsets(order).unwrap();
        let mut previous = vec![0.0f64; state.len()];
        let mut scaled = vec![0.0f64; order + 1];
        for idx in 1..values.len() {
            let delta = values[idx] - values[idx - 1];
            append_axis_segment(&mut state, &mut previous, &level_offsets, &mut scaled, 0, delta, order);
            append_axis_segment(&mut state, &mut previous, &level_offsets, &mut scaled, 1, delta, order);
        }
        state.into_iter().map(|value| value as f32).collect()
    }

    fn assert_close(left: &[f32], right: &[f32], tol: f32) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!(
                (left - right).abs() <= tol,
                "left={left} right={right} tol={tol}"
            );
        }
    }
}

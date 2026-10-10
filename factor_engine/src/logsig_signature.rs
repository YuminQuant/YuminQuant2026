use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{err, Result};

const DEFAULT_LOGSIG_THREADS: usize = 3;
static LOGSIG_THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();
static ORDER_TEN_KERNEL: OnceLock<Arc<SignatureKernel>> = OnceLock::new();

#[derive(Clone, Debug)]
struct LyndonBasis {
    words: Vec<(usize, usize)>,
    expansions: Vec<BTreeMap<usize, f32>>,
}

struct SignatureKernel {
    order: usize,
    width: usize,
    offsets: Vec<usize>,
    projection: Vec<(usize, Vec<(usize, f32)>)>,
    suffixes: [Vec<usize>; 2],
}

impl SignatureKernel {
    fn new(order: usize) -> Result<Self> {
        let width = tensor_signature_width(order)?;
        let offsets = level_offsets(order)?;
        let basis = lyndon_basis(order)?;
        let projection = basis
            .words
            .iter()
            .zip(&basis.expansions)
            .map(|(&(degree, word), expansion)| {
                (
                    offsets[degree] + word,
                    expansion
                        .iter()
                        .map(|(&w, &c)| (offsets[degree] + w, c))
                        .collect(),
                )
            })
            .collect();
        let mut suffixes = [vec![0; width], vec![0; width]];
        for axis in 0..2 {
            for level in 1..=order {
                for word in 0..1usize << level {
                    suffixes[axis][offsets[level] + word] =
                        repeated_axis_suffix_len(word, level, axis);
                }
            }
        }
        Ok(Self {
            order,
            width,
            offsets,
            projection,
            suffixes,
        })
    }
}

fn signature_kernel(order: usize) -> Result<Arc<SignatureKernel>> {
    if order == 10 {
        if let Some(kernel) = ORDER_TEN_KERNEL.get() {
            return Ok(Arc::clone(kernel));
        }
        let kernel = Arc::new(SignatureKernel::new(order)?);
        let _ = ORDER_TEN_KERNEL.set(Arc::clone(&kernel));
        return Ok(Arc::clone(ORDER_TEN_KERNEL.get().unwrap()));
    }
    Ok(Arc::new(SignatureKernel::new(order)?))
}

struct SignatureWorkspace {
    signature: Vec<f32>,
    scaled: Vec<f32>,
    powers: Vec<Vec<f32>>,
    residual: Vec<f32>,
}

impl SignatureWorkspace {
    fn new(kernel: &SignatureKernel) -> Self {
        Self {
            signature: vec![0.; kernel.width],
            scaled: vec![0.; kernel.order + 1],
            powers: vec![vec![0.; kernel.width]; kernel.order + 1],
            residual: vec![0.; kernel.width],
        }
    }

    fn compute(&mut self, volume: &[f32], kernel: &SignatureKernel, out: &mut [f32]) -> Result<()> {
        self.signature.fill(0.);
        let mut previous = clipped_log(volume[0])?;
        for &volume in &volume[1..] {
            let current = clipped_log(volume)?;
            let delta = current - previous;
            previous = current;
            if delta.abs() <= 1e-15 {
                continue;
            }
            append_axis_in_place(&mut self.signature, &mut self.scaled, kernel, 0, delta);
            append_axis_in_place(&mut self.signature, &mut self.scaled, kernel, 1, delta);
        }
        self.project(kernel, out);
        Ok(())
    }

    fn project(&mut self, kernel: &SignatureKernel, out: &mut [f32]) {
        let offsets = &kernel.offsets;
        self.powers[1].copy_from_slice(&self.signature);
        for power in 2..=kernel.order {
            let (before, after) = self.powers.split_at_mut(power);
            let previous = &before[power - 1];
            let current = &mut after[0];
            for level in power..=kernel.order {
                for word in 0..1usize << level {
                    let mut value = 0.;
                    for prefix_len in 1..=level - (power - 1) {
                        let suffix_len = level - prefix_len;
                        value += self.signature[offsets[prefix_len] + (word >> suffix_len)]
                            * previous[offsets[suffix_len] + (word & ((1usize << suffix_len) - 1))];
                    }
                    current[offsets[level] + word] = value;
                }
            }
        }
        self.residual.fill(0.);
        for power in 1..=kernel.order {
            let coefficient = if power % 2 == 1 {
                1. / power as f32
            } else {
                -1. / power as f32
            };
            for (dst, src) in self.residual.iter_mut().zip(&self.powers[power]) {
                *dst += coefficient * src;
            }
        }
        for (dst, (word, expansion)) in out.iter_mut().zip(&kernel.projection) {
            let coefficient = self.residual[*word];
            *dst = coefficient as f32;
            for &(index, value) in expansion {
                self.residual[index] -= coefficient * value;
            }
        }
    }
}

fn append_axis_in_place(
    signature: &mut [f32],
    scaled: &mut [f32],
    kernel: &SignatureKernel,
    axis: usize,
    delta: f32,
) {
    scaled[0] = 1.;
    for level in 1..=kernel.order {
        scaled[level] = scaled[level - 1] * delta / level as f32;
    }
    // Descending degree preserves all lower-degree prefixes until they have been consumed.
    for level in (1..=kernel.order).rev() {
        let offset = kernel.offsets[level];
        for word in (axis..1usize << level).step_by(2) {
            let mut value = signature[offset + word];
            for repeat in 1..=kernel.suffixes[axis][offset + word] {
                let prefix = if repeat == level {
                    1.
                } else {
                    signature[kernel.offsets[level - repeat] + (word >> repeat)]
                };
                value += prefix * scaled[repeat];
            }
            signature[offset + word] = value;
        }
    }
}

pub fn logsig_thread_count() -> usize {
    configured_logsig_thread_count(
        std::env::var("YQ_LOGSIG_THREADS").ok(),
        std::env::var("RAYON_NUM_THREADS").ok(),
    )
}

pub fn signature_width(order: usize) -> Result<usize> {
    logsignature_width(order)
}

pub fn logsignature_width(order: usize) -> Result<usize> {
    Ok(lyndon_words(order)?.len())
}

pub fn tensor_signature_width(order: usize) -> Result<usize> {
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

pub fn lyndon_degree_dimensions(order: usize) -> Result<Vec<usize>> {
    if order == 0 {
        return Err(err("logsignature order must be positive"));
    }
    let words = lyndon_words(order)?;
    let mut counts = vec![0usize; order];
    for (degree, _) in words {
        counts[degree - 1] += 1;
    }
    Ok(counts)
}

pub fn logsig_signature_batch_from_volume(
    volume: &[f32],
    rows: usize,
    cols: usize,
    order: usize,
) -> Result<Vec<f32>> {
    logsig_signature_batch_in_pool(volume, rows, cols, order, logsig_thread_pool())
}

pub fn logsig_signature_batch_in_pool(
    volume: &[f32],
    rows: usize,
    cols: usize,
    order: usize,
    pool: &ThreadPool,
) -> Result<Vec<f32>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if cols == 0 {
        return Err(err(
            "logsig signature volume matrix must have at least one column",
        ));
    }
    if volume.len() != rows.saturating_mul(cols) {
        return Err(err(format!(
            "logsig signature volume length {} does not match shape {rows}x{cols}",
            volume.len()
        )));
    }
    let kernel = signature_kernel(order)?;
    let logsig_width = kernel.projection.len();
    let mut output = vec![0.0f32; rows * logsig_width];
    pool.install(|| {
        output
            .par_chunks_mut(logsig_width * 64)
            .zip(volume.par_chunks(cols * 64))
            .try_for_each_init(
                || SignatureWorkspace::new(&kernel),
                |workspace, (out, rows)| -> Result<()> {
                    for (out, row) in out.chunks_mut(logsig_width).zip(rows.chunks(cols)) {
                        workspace.compute(row, &kernel, out)?;
                    }
                    Ok(())
                },
            )
    })?;
    Ok(output)
}

fn configured_logsig_thread_count(
    logsig_threads: Option<String>,
    rayon_threads: Option<String>,
) -> usize {
    logsig_threads
        .or(rayon_threads)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LOGSIG_THREADS)
}

fn logsig_thread_pool() -> &'static ThreadPool {
    LOGSIG_THREAD_POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(logsig_thread_count())
            .thread_name(|idx| format!("yq-logsig-{idx}"))
            .build()
            .expect("failed to build logsig signature thread pool")
    })
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

#[cfg(test)]
fn compute_row(
    volume: &[f32],
    order: usize,
    tensor_width: usize,
    level_offsets: &[usize],
    basis: &LyndonBasis,
    out: &mut [f32],
) -> Result<()> {
    let mut signature = vec![0.0f32; tensor_width];
    let mut previous = vec![0.0f32; tensor_width];
    let mut scaled = vec![0.0f32; order + 1];
    let mut previous_log = clipped_log(volume[0])?;
    for value in volume.iter().copied().skip(1) {
        let current_log = clipped_log(value)?;
        let delta = current_log - previous_log;
        previous_log = current_log;
        if delta.abs() <= 1e-15 {
            continue;
        }
        append_axis_segment(
            &mut signature,
            &mut previous,
            level_offsets,
            &mut scaled,
            0,
            delta,
            order,
        );
        append_axis_segment(
            &mut signature,
            &mut previous,
            level_offsets,
            &mut scaled,
            1,
            delta,
            order,
        );
    }
    let tensor_log = tensor_log(&signature, order, level_offsets);
    let logsig = project_tensor_log_to_lyndon(&tensor_log, order, level_offsets, basis);
    for (dst, src) in out.iter_mut().zip(logsig) {
        *dst = src as f32;
    }
    Ok(())
}

fn clipped_log(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(err("logsig signature volume contains non-finite value"));
    }
    Ok(value.max(1.0).ln())
}

#[cfg(test)]
fn append_axis_segment(
    levels: &mut [f32],
    previous: &mut [f32],
    level_offsets: &[usize],
    scaled: &mut [f32],
    axis: usize,
    delta: f32,
    order: usize,
) {
    previous.copy_from_slice(levels);
    scaled[0] = 1.0;
    for level in 1..=order {
        scaled[level] = scaled[level - 1] * delta / level as f32;
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
fn tensor_log(signature: &[f32], order: usize, level_offsets: &[usize]) -> Vec<f32> {
    let mut powers = vec![vec![0.0f32; signature.len()]; order + 1];
    powers[1].copy_from_slice(signature);
    for power in 2..=order {
        for level in power..=order {
            let level_width = 1usize << level;
            for word in 0..level_width {
                let mut value = 0.0;
                for prefix_len in 1..=(level - (power - 1)) {
                    let suffix_len = level - prefix_len;
                    let prefix_word = word >> suffix_len;
                    let suffix_mask = (1usize << suffix_len) - 1;
                    let suffix_word = word & suffix_mask;
                    value += signature[level_offsets[prefix_len] + prefix_word]
                        * powers[power - 1][level_offsets[suffix_len] + suffix_word];
                }
                powers[power][level_offsets[level] + word] = value;
            }
        }
    }

    let mut output = vec![0.0f32; signature.len()];
    for (power, values) in powers.iter().enumerate().take(order + 1).skip(1) {
        let coefficient = if power % 2 == 1 {
            1.0 / power as f32
        } else {
            -1.0 / power as f32
        };
        for (dst, src) in output.iter_mut().zip(values) {
            *dst += coefficient * src;
        }
    }
    output
}

fn lyndon_words(order: usize) -> Result<Vec<(usize, usize)>> {
    if order == 0 {
        return Err(err("logsignature order must be positive"));
    }
    let mut output = Vec::new();
    for length in 1..=order {
        let width = 1usize
            .checked_shl(length as u32)
            .ok_or_else(|| err(format!("logsignature order {order} is too large")))?;
        for word in 0..width {
            if is_lyndon(word, length) {
                output.push((length, word));
            }
        }
    }
    Ok(output)
}

fn is_lyndon(word: usize, length: usize) -> bool {
    if length == 1 {
        return true;
    }
    (1..length).all(|shift| word < rotate_left_word(word, length, shift))
}

fn rotate_left_word(word: usize, length: usize, shift: usize) -> usize {
    let mut output = 0usize;
    for pos in 0..length {
        let source_pos = (pos + shift) % length;
        output = (output << 1) | letter(word, length, source_pos);
    }
    output
}

fn letter(word: usize, length: usize, pos: usize) -> usize {
    (word >> (length - 1 - pos)) & 1
}

fn lyndon_basis(order: usize) -> Result<LyndonBasis> {
    let words = lyndon_words(order)?;
    let word_set = words
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut expansions_by_word: HashMap<(usize, usize), BTreeMap<usize, f32>> = HashMap::new();
    let mut expansions = Vec::with_capacity(words.len());

    for (length, word) in words.iter().copied() {
        let expansion = if length == 1 {
            BTreeMap::from([(word, 1.0)])
        } else {
            let (prefix_len, prefix_word, suffix_len, suffix_word) =
                standard_factorization(length, word, &word_set)?;
            let left = expansions_by_word
                .get(&(prefix_len, prefix_word))
                .ok_or_else(|| err("missing Lyndon prefix expansion"))?;
            let right = expansions_by_word
                .get(&(suffix_len, suffix_word))
                .ok_or_else(|| err("missing Lyndon suffix expansion"))?;
            bracket_expansion(left, prefix_len, right, suffix_len)
        };
        let leading = expansion.get(&word).copied().unwrap_or(0.0);
        if (leading - 1.0).abs() > 1e-10 {
            return Err(err(format!(
                "invalid Lyndon basis expansion for word {word}: leading coefficient {leading}"
            )));
        }
        expansions_by_word.insert((length, word), expansion.clone());
        expansions.push(expansion);
    }

    Ok(LyndonBasis { words, expansions })
}

fn standard_factorization(
    length: usize,
    word: usize,
    word_set: &std::collections::HashSet<(usize, usize)>,
) -> Result<(usize, usize, usize, usize)> {
    for suffix_len in (1..length).rev() {
        let suffix_mask = (1usize << suffix_len) - 1;
        let suffix_word = word & suffix_mask;
        if word_set.contains(&(suffix_len, suffix_word)) {
            let prefix_len = length - suffix_len;
            let prefix_word = word >> suffix_len;
            if word_set.contains(&(prefix_len, prefix_word)) {
                return Ok((prefix_len, prefix_word, suffix_len, suffix_word));
            }
        }
    }
    Err(err(format!(
        "could not factor Lyndon word length={length} word={word}"
    )))
}

fn bracket_expansion(
    left: &BTreeMap<usize, f32>,
    left_len: usize,
    right: &BTreeMap<usize, f32>,
    right_len: usize,
) -> BTreeMap<usize, f32> {
    let mut output = BTreeMap::<usize, f32>::new();
    for (left_word, left_coeff) in left {
        for (right_word, right_coeff) in right {
            let coeff = left_coeff * right_coeff;
            let lr = (left_word << right_len) | right_word;
            let rl = (right_word << left_len) | left_word;
            *output.entry(lr).or_default() += coeff;
            *output.entry(rl).or_default() -= coeff;
        }
    }
    output.retain(|_, value| value.abs() > 1e-14);
    output
}

#[cfg(test)]
fn project_tensor_log_to_lyndon(
    tensor_log: &[f32],
    order: usize,
    level_offsets: &[usize],
    basis: &LyndonBasis,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(basis.words.len());
    let mut start = 0usize;
    while start < basis.words.len() {
        let degree = basis.words[start].0;
        let end = basis.words[start..]
            .iter()
            .position(|(candidate_degree, _)| *candidate_degree != degree)
            .map(|offset| start + offset)
            .unwrap_or(basis.words.len());
        let mut residual =
            tensor_log[level_offsets[degree]..level_offsets[degree] + (1usize << degree)].to_vec();
        for idx in start..end {
            let word = basis.words[idx].1;
            let coefficient = residual[word];
            output.push(coefficient);
            for (expanded_word, expanded_coeff) in &basis.expansions[idx] {
                residual[*expanded_word] -= coefficient * expanded_coeff;
            }
        }
        start = end;
    }
    debug_assert_eq!(output.len(), logsignature_width(order).unwrap());
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_kernel_matches_reference_bitwise() {
        for order in [1, 2, 5, 10] {
            let width = tensor_signature_width(order).unwrap();
            let offsets = level_offsets(order).unwrap();
            let basis = lyndon_basis(order).unwrap();
            let kernel = signature_kernel(order).unwrap();
            let mut workspace = SignatureWorkspace::new(&kernel);
            for seed in 0..6 {
                let volume: Vec<_> = (0..960)
                    .map(|i| match seed {
                        0 => 0.,
                        1 => 100.,
                        _ => (((i * 173 + seed * 31) % 1901) as f32 + 1.) * 19.7,
                    })
                    .collect();
                let mut reference = vec![0.; basis.words.len()];
                let mut actual = reference.clone();
                compute_row(&volume, order, width, &offsets, &basis, &mut reference).unwrap();
                workspace.compute(&volume, &kernel, &mut actual).unwrap();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "order={order} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn batch_kernel_preserves_rows_across_partial_chunks() {
        let volume: Vec<_> = (0..67 * 8).map(|i| (i % 23 + 1) as f32).collect();
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let actual = logsig_signature_batch_in_pool(&volume, 67, 8, 5, &pool).unwrap();
        let basis = lyndon_basis(5).unwrap();
        let offsets = level_offsets(5).unwrap();
        for (row, out) in volume.chunks(8).zip(actual.chunks(basis.words.len())) {
            let mut expected = vec![0.; basis.words.len()];
            compute_row(
                row,
                5,
                tensor_signature_width(5).unwrap(),
                &offsets,
                &basis,
                &mut expected,
            )
            .unwrap();
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn width_matches_order_ten_lyndon_dimension() {
        assert_eq!(
            lyndon_degree_dimensions(10).unwrap(),
            vec![2, 1, 2, 3, 6, 9, 18, 30, 56, 99]
        );
        assert_eq!(logsignature_width(10).unwrap(), 226);
        assert_eq!(signature_width(10).unwrap(), 226);
        assert_eq!(tensor_signature_width(10).unwrap(), 2046);
    }

    #[test]
    fn default_logsig_thread_count_is_small_and_overridable() {
        assert_eq!(configured_logsig_thread_count(None, None), 3);
        assert_eq!(
            configured_logsig_thread_count(Some("2".to_string()), None),
            2
        );
        assert_eq!(
            configured_logsig_thread_count(None, Some("4".to_string())),
            4
        );
        assert_eq!(
            configured_logsig_thread_count(Some("0".to_string()), Some("0".to_string())),
            3
        );
    }

    #[test]
    fn zero_volume_is_clipped_before_log() {
        let with_zero = logsig_signature_batch_from_volume(&[0.0, 10.0], 1, 2, 3).unwrap();
        let clipped = logsig_signature_batch_from_volume(&[1.0, 10.0], 1, 2, 3).unwrap();
        assert_close(&with_zero, &clipped, 1e-6);
    }

    #[test]
    fn two_point_lead_lag_logsignature_order_two_matches_bch() {
        let actual = logsig_signature_batch_from_volume(&[1.0, 10.0], 1, 2, 2).unwrap();
        let delta = 10.0f32.ln() as f32;
        assert_eq!(actual.len(), 3);
        assert_close(&actual, &[delta, delta, 0.5 * delta * delta], 1e-6);
    }

    #[test]
    fn lyndon_words_are_degree_then_lexicographic() {
        let words = lyndon_words(4).unwrap();
        assert_eq!(
            words,
            vec![
                (1, 0b0),
                (1, 0b1),
                (2, 0b01),
                (3, 0b001),
                (3, 0b011),
                (4, 0b0001),
                (4, 0b0011),
                (4, 0b0111),
            ]
        );
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
        let error = logsig_signature_batch_from_volume(&[1.0, f32::NAN], 1, 2, 2).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
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

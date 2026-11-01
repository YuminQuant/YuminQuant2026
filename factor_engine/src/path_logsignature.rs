//! Multivariate piecewise-linear paths in the standard Lyndon bracket basis.
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::{err, Result};

pub struct PathLogSignature {
    pub dimension: usize,
    pub order: usize,
    powers: Vec<usize>,
    offsets: Vec<usize>,
    width: usize,
    projection: Vec<(usize, Vec<(usize, f64)>)>,
}

impl PathLogSignature {
    pub fn new(dimension: usize, order: usize) -> Result<Self> {
        if dimension == 0 || order == 0 {
            return Err(err(
                "path logsignature requires positive dimension and order",
            ));
        }
        let mut powers = vec![1usize];
        let mut offsets = vec![0usize];
        let mut width = 0usize;
        for degree in 1..=order {
            let power = powers[degree - 1]
                .checked_mul(dimension)
                .filter(|&n| n <= 1_000_000)
                .ok_or_else(|| err("path tensor is too large"))?;
            powers.push(power);
            offsets.push(width);
            width += power;
        }
        let words = lyndon_words(dimension, order);
        let set: HashSet<_> = words.iter().cloned().collect();
        let mut expansions = HashMap::<Vec<usize>, BTreeMap<usize, f64>>::new();
        let mut projection = Vec::with_capacity(words.len());
        for word in words {
            let degree = word.len();
            let encoded = encode(&word, dimension);
            let expansion = if degree == 1 {
                BTreeMap::from([(encoded, 1.0)])
            } else {
                let split = (1..degree)
                    .find(|&split| set.contains(&word[split..]))
                    .ok_or_else(|| err("missing Lyndon standard factorization"))?;
                let left = expansions
                    .get(&word[..split])
                    .ok_or_else(|| err("missing Lyndon prefix"))?;
                let right = expansions
                    .get(&word[split..])
                    .ok_or_else(|| err("missing Lyndon suffix"))?;
                let mut bracket = BTreeMap::<usize, f64>::new();
                for (&a, &x) in left {
                    for (&b, &y) in right {
                        *bracket.entry(a * powers[degree - split] + b).or_default() += x * y;
                        *bracket.entry(b * powers[split] + a).or_default() -= x * y;
                    }
                }
                bracket.retain(|_, value| *value != 0.0);
                bracket
            };
            if expansion.get(&encoded) != Some(&1.0) {
                return Err(err("invalid Lyndon leading coefficient"));
            }
            projection.push((
                offsets[degree] + encoded,
                expansion
                    .iter()
                    .map(|(&w, &c)| (offsets[degree] + w, c))
                    .collect(),
            ));
            expansions.insert(word, expansion);
        }
        Ok(Self {
            dimension,
            order,
            powers,
            offsets,
            width,
            projection,
        })
    }

    pub fn log_width(&self) -> usize {
        self.projection.len()
    }

    /// Flattened successive increments; no basepoint augmentation is added.
    pub fn signature(&self, increments: &[f64]) -> Result<Vec<f64>> {
        if increments.len() % self.dimension != 0 || !increments.iter().all(|v| v.is_finite()) {
            return Err(err("invalid path increments"));
        }
        let mut signature = vec![0.0; self.width];
        let mut exponential = vec![0.0; self.width];
        for delta in increments.chunks_exact(self.dimension) {
            exponential[..self.dimension].copy_from_slice(delta);
            for degree in 2..=self.order {
                for word in 0..self.powers[degree] {
                    exponential[self.offsets[degree] + word] = exponential
                        [self.offsets[degree - 1] + word / self.dimension]
                        * delta[word % self.dimension]
                        / degree as f64;
                }
            }
            for degree in (1..=self.order).rev() {
                for word in 0..self.powers[degree] {
                    let mut value = exponential[self.offsets[degree] + word];
                    for split in 1..degree {
                        let suffix = degree - split;
                        value += signature[self.offsets[split] + word / self.powers[suffix]]
                            * exponential[self.offsets[suffix] + word % self.powers[suffix]];
                    }
                    signature[self.offsets[degree] + word] += value;
                }
            }
        }
        Ok(signature)
    }

    /// Chen product for consecutive path segments, scalar terms both equal 1.
    pub fn concatenate(&self, left: &[f64], right: &[f64]) -> Vec<f64> {
        self.product(left, right, true)
    }

    fn product(&self, left: &[f64], right: &[f64], identities: bool) -> Vec<f64> {
        let mut output = vec![0.0; self.width];
        for degree in 1..=self.order {
            for word in 0..self.powers[degree] {
                let index = self.offsets[degree] + word;
                let mut value = if identities {
                    left[index] + right[index]
                } else {
                    0.0
                };
                for split in 1..degree {
                    let suffix = degree - split;
                    value += left[self.offsets[split] + word / self.powers[suffix]]
                        * right[self.offsets[suffix] + word % self.powers[suffix]];
                }
                output[index] = value;
            }
        }
        output
    }

    pub fn project(&self, signature: &[f64]) -> Vec<f64> {
        let mut residual = signature.to_vec();
        let mut power = signature.to_vec();
        for exponent in 2..=self.order {
            power = self.product(&power, signature, false);
            let coefficient = if exponent % 2 == 0 { -1.0 } else { 1.0 } / exponent as f64;
            for (dst, value) in residual.iter_mut().zip(&power) {
                *dst += coefficient * value;
            }
        }
        let mut output = Vec::with_capacity(self.log_width());
        for (word, expansion) in &self.projection {
            let coefficient = residual[*word];
            output.push(coefficient);
            for &(index, value) in expansion {
                residual[index] -= coefficient * value;
            }
        }
        output
    }
}

fn encode(word: &[usize], dimension: usize) -> usize {
    word.iter()
        .fold(0, |value, &letter| value * dimension + letter)
}

pub fn lyndon_words(dimension: usize, order: usize) -> Vec<Vec<usize>> {
    let mut output = Vec::new();
    for degree in 1..=order {
        for mut encoded in 0..dimension.pow(degree as u32) {
            let mut word = vec![0; degree];
            for letter in word.iter_mut().rev() {
                *letter = encoded % dimension;
                encoded /= dimension;
            }
            if (1..degree).all(|shift| {
                word.iter()
                    .cmp(word[shift..].iter().chain(&word[..shift]))
                    .is_lt()
            }) {
                output.push(word);
            }
        }
    }
    output
}

//! Small daily regressions. No stock-by-stock or minute-by-minute dense matrices.

pub fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

pub fn corr(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let a = pairs.iter().map(|x| x.0).sum::<f64>() / n;
    let b = pairs.iter().map(|x| x.1).sum::<f64>() / n;
    let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        aa += (x - a).powi(2);
        bb += (y - b).powi(2);
        ab += (x - a) * (y - b);
    }
    if aa <= 0.0 || bb <= 0.0 {
        return None;
    }
    let r = ab / (aa * bb).sqrt();
    r.is_finite().then(|| r.clamp(-1.0, 1.0))
}

fn solve(mut a: [[f64; 5]; 5], mut b: [f64; 5], k: usize) -> Option<[f64; 5]> {
    for j in 0..k {
        let pivot = (j..k).max_by(|&i, &l| a[i][j].abs().total_cmp(&a[l][j].abs()))?;
        if !a[pivot][j].is_finite() || a[pivot][j].abs() < 1e-28 {
            return None;
        }
        a.swap(j, pivot);
        b.swap(j, pivot);
        let d = a[j][j];
        for l in j..k {
            a[j][l] /= d;
        }
        b[j] /= d;
        for i in 0..k {
            if i == j {
                continue;
            }
            let v = a[i][j];
            for l in j..k {
                a[i][l] -= v * a[j][l];
            }
            b[i] -= v * b[j];
        }
    }
    b[..k].iter().all(|v| v.is_finite()).then_some(b)
}

fn dot(x: &[f64; 5], b: &[f64; 5], k: usize) -> f64 {
    (0..k).map(|j| x[j] * b[j]).sum()
}

// Reorthogonalization removes constant/duplicate regressors without changing fitted values.
fn orthogonal_rows(x: &[[f64; 5]]) -> (Vec<[f64; 5]>, usize) {
    let mut rows = vec![[0.0; 5]; x.len()];
    let mut k = 0;
    for j in 0..5 {
        let mut col: Vec<_> = x.iter().map(|r| r[j]).collect();
        let original = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if original == 0.0 {
            continue;
        }
        for v in &mut col {
            *v /= original;
        }
        for _ in 0..2 {
            for l in 0..k {
                let projection: f64 = col.iter().zip(&rows).map(|(v, r)| v * r[l]).sum();
                for (v, r) in col.iter_mut().zip(&rows) {
                    *v -= projection * r[l];
                }
            }
        }
        let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-10 {
            continue;
        }
        for (r, v) in rows.iter_mut().zip(col) {
            r[k] = v / norm;
        }
        k += 1;
    }
    (rows, k)
}

struct Direction {
    b: [f64; 5],
    u: Vec<f64>,
    plus: Vec<f64>,
    minus: Vec<f64>,
}

impl Direction {
    fn new(n: usize) -> Self {
        Self {
            b: [0.0; 5],
            u: vec![0.0; n],
            plus: vec![0.0; n],
            minus: vec![0.0; n],
        }
    }
}

// Record the original Gauss-Jordan operations so both RHSs use identical arithmetic.
struct Elimination {
    pivots: [usize; 5],
    divisors: [f64; 5],
    multipliers: [[f64; 5]; 5],
    k: usize,
}
impl Elimination {
    fn new(mut a: [[f64; 5]; 5], k: usize) -> Option<Self> {
        let mut result = Self {
            pivots: [0; 5],
            divisors: [0.0; 5],
            multipliers: [[0.0; 5]; 5],
            k,
        };
        for j in 0..k {
            let pivot = (j..k).max_by(|&i, &l| a[i][j].abs().total_cmp(&a[l][j].abs()))?;
            if !a[pivot][j].is_finite() || a[pivot][j].abs() < 1e-28 {
                return None;
            }
            result.pivots[j] = pivot;
            a.swap(j, pivot);
            let d = a[j][j];
            result.divisors[j] = d;
            for l in j..k {
                a[j][l] /= d;
            }
            for i in 0..k {
                if i == j {
                    continue;
                }
                let v = a[i][j];
                result.multipliers[j][i] = v;
                for l in j..k {
                    a[i][l] -= v * a[j][l];
                }
            }
        }
        Some(result)
    }
    fn solve(&self, mut b: [f64; 5]) -> Option<[f64; 5]> {
        for j in 0..self.k {
            b.swap(j, self.pivots[j]);
            b[j] /= self.divisors[j];
            for i in 0..self.k {
                if i != j {
                    b[i] -= self.multipliers[j][i] * b[j];
                }
            }
        }
        b[..self.k].iter().all(|v| v.is_finite()).then_some(b)
    }
}

struct DirectionWorkspace {
    ds: Vec<f64>,
    rhs: Vec<f64>,
    tp: Vec<f64>,
    tm: Vec<f64>,
}
impl DirectionWorkspace {
    fn new(n: usize) -> Self {
        Self {
            ds: vec![0.0; n],
            rhs: vec![0.0; n],
            tp: vec![0.0; n],
            tm: vec![0.0; n],
        }
    }
    fn prepare(
        &mut self,
        x: &[[f64; 5]],
        k: usize,
        u: &[f64],
        plus: &[f64],
        minus: &[f64],
        q: f64,
    ) -> Option<Elimination> {
        let mut gram = [[0.0; 5]; 5];
        for i in 0..x.len() {
            self.ds[i] = plus[i] / (q - u[i]) + minus[i] / (1.0 - q + u[i]);
            for j in 0..k {
                for l in 0..k {
                    gram[j][l] += x[i][j] * x[i][l] / self.ds[i];
                }
            }
        }
        Elimination::new(gram, k)
    }
}

fn direction(
    x: &[[f64; 5]],
    y: &[f64],
    k: usize,
    b: &[f64; 5],
    u: &[f64],
    plus: &[f64],
    minus: &[f64],
    q: f64,
    target: f64,
    affine: Option<&Direction>,
    work: &mut DirectionWorkspace,
    elimination: &Elimination,
    d: &mut Direction,
) -> Option<()> {
    let n = y.len();
    let mut rhs_b = [0.0; 5];
    let DirectionWorkspace { ds, rhs, tp, tm } = work;
    for i in 0..n {
        let sp = q - u[i];
        let sm = 1.0 - q + u[i];
        tp[i] = target - plus[i] * sp;
        tm[i] = target - minus[i] * sm;
        if let Some(d) = affine {
            tp[i] += d.plus[i] * d.u[i];
            tm[i] -= d.minus[i] * d.u[i];
        }
        rhs[i] = y[i] - dot(&x[i], b, k) - plus[i] + minus[i] - tp[i] / sp + tm[i] / sm;
        for j in 0..k {
            rhs_b[j] += x[i][j] * (rhs[i] / ds[i] + u[i]);
        }
    }
    let db = elimination.solve(rhs_b)?;
    d.b = db;
    for i in 0..n {
        d.u[i] = (rhs[i] - dot(&x[i], &db, k)) / ds[i];
        d.plus[i] = (tp[i] + plus[i] * d.u[i]) / (q - u[i]);
        d.minus[i] = (tm[i] - minus[i] * d.u[i]) / (1.0 - q + u[i]);
    }
    Some(())
}

fn step(d: &Direction, plus: &[f64], minus: &[f64], u: &[f64], q: f64) -> f64 {
    let mut a: f64 = 1.0;
    for i in 0..u.len() {
        if d.plus[i] < 0.0 {
            a = a.min(-plus[i] / d.plus[i]);
        }
        if d.minus[i] < 0.0 {
            a = a.min(-minus[i] / d.minus[i]);
        }
        if d.u[i] > 0.0 {
            a = a.min((q - u[i]) / d.u[i]);
        }
        if d.u[i] < 0.0 {
            a = a.min(-(1.0 - q + u[i]) / d.u[i]);
        }
    }
    a
}

/// Quantile-loss LP, primal-dual predictor/corrector; never return an unconverged fit.
pub fn quantile_residual(x: &[[f64; 5]], y: &[f64], q: f64) -> Option<Vec<f64>> {
    if y.len() < 20 || x.len() != y.len() || !(0.0..1.0).contains(&q) {
        return None;
    }
    if !y.iter().chain(x.iter().flatten()).all(|v| v.is_finite()) {
        return None;
    }
    let scale = y.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let y: Vec<_> = y.iter().map(|v| v / scale).collect();
    let (x, k) = orthogonal_rows(x);
    if k == 0 {
        return None;
    }
    let n = y.len();
    let mut b = [0.0; 5];
    let mut u = vec![0.0; n];
    let mut plus: Vec<_> = y.iter().map(|v| v.max(0.0) + 1.0).collect();
    let mut minus: Vec<_> = y.iter().map(|v| (-v).max(0.0) + 1.0).collect();
    let mut work = DirectionWorkspace::new(n);
    let mut affine = Direction::new(n);
    let mut d = Direction::new(n);
    for _ in 0..80 {
        let mu = (0..n)
            .map(|i| plus[i] * (q - u[i]) + minus[i] * (1.0 - q + u[i]))
            .sum::<f64>()
            / (2 * n) as f64;
        let primal = (0..n)
            .map(|i| (dot(&x[i], &b, k) + plus[i] - minus[i] - y[i]).abs())
            .fold(0.0, f64::max);
        let dual = (0..k)
            .map(|j| x.iter().zip(&u).map(|(r, v)| r[j] * v).sum::<f64>().abs())
            .fold(0.0, f64::max);
        if mu < 1e-9 && primal < 1e-8 && dual < 1e-8 {
            let residual: Vec<_> = (0..n).map(|i| (y[i] - dot(&x[i], &b, k)) * scale).collect();
            return residual.iter().all(|v| v.is_finite()).then_some(residual);
        }
        let elimination = work.prepare(&x, k, &u, &plus, &minus, q)?;
        direction(
            &x,
            &y,
            k,
            &b,
            &u,
            &plus,
            &minus,
            q,
            0.0,
            None,
            &mut work,
            &elimination,
            &mut affine,
        )?;
        let a = step(&affine, &plus, &minus, &u, q);
        let mu_aff = (0..n)
            .map(|i| {
                (plus[i] + a * affine.plus[i]) * (q - u[i] - a * affine.u[i])
                    + (minus[i] + a * affine.minus[i]) * (1.0 - q + u[i] + a * affine.u[i])
            })
            .sum::<f64>()
            / (2 * n) as f64;
        let sigma = (mu_aff / mu).clamp(0.0, 1.0).powi(3);
        direction(
            &x,
            &y,
            k,
            &b,
            &u,
            &plus,
            &minus,
            q,
            sigma * mu,
            Some(&affine),
            &mut work,
            &elimination,
            &mut d,
        )?;
        let a = 0.995 * step(&d, &plus, &minus, &u, q);
        if !a.is_finite() || a <= 0.0 {
            return None;
        }
        for j in 0..k {
            b[j] += a * d.b[j];
        }
        for i in 0..n {
            u[i] += a * d.u[i];
            plus[i] += a * d.plus[i];
            minus[i] += a * d.minus[i];
        }
    }
    None
}

/// OLS on [1, |market|, 100*market^2], with scaled coordinates and ordinary R^2.
pub fn csad_fit(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < 4 {
        return None;
    }
    let scale = pairs.iter().fold(0.0_f64, |s, p| s.max(p.0.abs()));
    if scale <= 0.0 {
        return None;
    }
    let mut a = [[0.0; 5]; 5];
    let mut b = [0.0; 5];
    for &(m, y) in pairs {
        let x = [1.0, m.abs() / scale, (m / scale).powi(2)];
        for j in 0..3 {
            b[j] += x[j] * y;
            for l in 0..3 {
                a[j][l] += x[j] * x[l];
            }
        }
    }
    // Reject rank-deficient regressors rather than fabricating a coefficient.
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 * a[0][0] * a[1][1] * a[2][2] {
        return None;
    }
    let b = solve(a, b, 3)?;
    let avg = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64;
    let total = pairs.iter().map(|p| (p.1 - avg).powi(2)).sum::<f64>();
    if total <= 0.0 {
        return None;
    }
    let error = pairs
        .iter()
        .map(|&(m, y)| (y - b[0] - b[1] * m.abs() / scale - b[2] * (m / scale).powi(2)).powi(2))
        .sum::<f64>();
    let result = (
        b[2] / (100.0 * scale * scale),
        (1.0 - error / total).clamp(0.0, 1.0),
    );
    (result.0.is_finite() && result.1.is_finite()).then_some(result)
}

/// Diagonal quadratic columns interpolate their selected rows. Optimize the slope
/// on the complement, using exact SSE changes for every one-in/one-out exchange.
pub fn sparse_gamma(pairs: &[(f64, f64)]) -> Option<Vec<f64>> {
    let n = pairs.len();
    if n < 10 {
        return None;
    }
    let scale = pairs.iter().fold(0.0_f64, |s, p| s.max(p.0.abs()));
    if scale <= 0.0 {
        return None;
    }
    let x: Vec<_> = pairs.iter().map(|p| p.0.abs() / scale).collect();
    let s = (n / 10).min(24);
    let terms: Vec<_> = x
        .iter()
        .zip(pairs)
        .map(|(&x, &(_, y))| [x * x, x * y, y * y])
        .collect();
    let mut totals = [0.0; 3];
    for v in &terms {
        for j in 0..3 {
            totals[j] += v[j];
        }
    }
    if totals[0] <= 1e-20 {
        return None;
    }
    let beta = totals[1] / totals[0];
    let mut candidates: Vec<_> = (0..n).filter(|&i| x[i] > 0.0).collect();
    candidates.sort_by(|&i, &j| {
        (pairs[j].1 - beta * x[j])
            .abs()
            .total_cmp(&(pairs[i].1 - beta * x[i]).abs())
            .then(i.cmp(&j))
    });
    let mut selected = vec![false; n];
    for &i in candidates.iter().take(s) {
        selected[i] = true;
        for j in 0..3 {
            totals[j] -= terms[i][j];
        }
    }
    let objective = |a: [f64; 3]| {
        if a[0] > 1e-20 {
            (a[2] - a[1] * a[1] / a[0]).max(0.0)
        } else {
            f64::INFINITY
        }
    };
    for _ in 0..n {
        let current = objective(totals);
        if !current.is_finite() {
            return None;
        }
        let mut best = current;
        let mut swap = None;
        for out in 0..n {
            if !selected[out] {
                continue;
            }
            for inside in 0..n {
                if selected[inside] || x[inside] == 0.0 {
                    continue;
                }
                let next = std::array::from_fn(|j| totals[j] + terms[out][j] - terms[inside][j]);
                let loss = objective(next);
                if loss < best - 1e-12 * current.max(1e-20) {
                    best = loss;
                    swap = Some((out, inside, next));
                }
            }
        }
        if let Some((out, inside, next)) = swap {
            selected[out] = false;
            selected[inside] = true;
            totals = next;
        } else {
            let beta = totals[1] / totals[0];
            let gamma: Vec<_> = (0..n)
                .map(|i| {
                    if selected[i] {
                        (pairs[i].1 - beta * x[i]) / pairs[i].0.powi(2)
                    } else {
                        0.0
                    }
                })
                .collect();
            return gamma.iter().all(|v| v.is_finite()).then_some(gamma);
        }
    }
    None
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Counts {
    pub n: usize,
    pub h: usize,
}

pub fn onset(gamma: &[Option<f64>], residual: &[[Option<f64>; 2]], high: bool) -> [Counts; 2] {
    let mut result = [Counts::default(); 2];
    for t in 0..gamma.len().saturating_sub(5) {
        let (Some(g), Some(low), Some(upper)) = (gamma[t], residual[t][0], residual[t][1]) else {
            continue;
        };
        if g > 0.0 || low < upper || gamma[t + 1..=t + 5].iter().any(Option::is_none) {
            continue;
        }
        let event = if high { upper > 0.0 } else { low < 0.0 };
        let group = if event { 0 } else { 1 };
        result[group].n += 1;
        result[group].h += usize::from(gamma[t + 1..=t + 5].iter().any(|g| g.unwrap() > 0.0));
    }
    result
}

pub fn basic_onset(c: [Counts; 2]) -> Option<f64> {
    if c.iter().any(|c| c.n == 0) {
        return None;
    }
    Some(
        ((c[1].h as f64 + 0.5) / (c[1].n as f64 + 1.0)).ln()
            - ((c[0].h as f64 + 0.5) / (c[0].n as f64 + 1.0)).ln(),
    )
}

pub fn prior(c: &[Counts]) -> Option<(f64, f64)> {
    let n = c.iter().map(|c| c.n).sum::<usize>();
    if n == 0 {
        return None;
    }
    let h = c.iter().map(|c| c.h).sum::<usize>();
    let m = (h as f64 + 0.5) / (n as f64 + 1.0);
    let count = c.iter().filter(|c| c.n > 0).count();
    let observed = c
        .iter()
        .filter(|c| c.n > 0)
        .map(|c| c.n as f64 * (c.h as f64 / c.n as f64 - m).powi(2))
        .sum::<f64>()
        / n as f64;
    let noise = m * (1.0 - m) * count as f64 / n as f64;
    let k = (m * (1.0 - m) / (observed - noise).max(1e-8) - 1.0).clamp(5.0, 200.0);
    Some((m, k))
}

pub fn bayes_onset(c: [Counts; 2], p: [(f64, f64); 2]) -> Option<f64> {
    if c[0].n < 3 || c[1].n < 5 {
        return None;
    }
    let prob = |i: usize| (c[i].h as f64 + p[i].0 * p[i].1) / (c[i].n as f64 + p[i].1);
    Some(prob(1).ln() - prob(0).ln())
}

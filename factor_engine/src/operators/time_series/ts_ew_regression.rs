use super::utils;

pub fn ts_ew_regression_beta_residual_sigma(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
    half_life: f64,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let (_alpha, beta, residual_sigma) =
        ts_ew_regression_alpha_beta_residual_sigma(y, x, window, min_periods, half_life);
    (beta, residual_sigma)
}

pub fn ts_ew_regression_alpha_beta_residual_sigma(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
    half_life: f64,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = y.len().min(x.len());
    let mut alpha = vec![None; len];
    let mut beta = vec![None; len];
    let mut residual_sigma = vec![None; len];
    if window == 0 || min_periods == 0 || min_periods > window || half_life <= 0.0 {
        return (alpha, beta, residual_sigma);
    }
    let decay = 0.5_f64.powf(1.0 / half_life);
    let expired_weight = decay.powi(window as i32);

    let mut valid_count = 0usize;
    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_yy = 0.0;

    for idx in 0..len {
        sum_w *= decay;
        sum_x *= decay;
        sum_y *= decay;
        sum_xx *= decay;
        sum_xy *= decay;
        sum_yy *= decay;

        if let (Some(y_value), Some(x_value)) =
            (utils::valid_value(y[idx]), utils::valid_value(x[idx]))
        {
            valid_count += 1;
            sum_w += 1.0;
            sum_x += x_value;
            sum_y += y_value;
            sum_xx += x_value * x_value;
            sum_xy += x_value * y_value;
            sum_yy += y_value * y_value;
        }

        if idx >= window {
            if let (Some(y_value), Some(x_value)) = (
                utils::valid_value(y[idx - window]),
                utils::valid_value(x[idx - window]),
            ) {
                valid_count -= 1;
                sum_w -= expired_weight;
                sum_x -= expired_weight * x_value;
                sum_y -= expired_weight * y_value;
                sum_xx -= expired_weight * x_value * x_value;
                sum_xy -= expired_weight * x_value * y_value;
                sum_yy -= expired_weight * y_value * y_value;
            }
        }

        if valid_count < min_periods || sum_w <= f64::EPSILON {
            continue;
        }
        let sxx = sum_xx - sum_x * sum_x / sum_w;
        if sxx <= f64::EPSILON {
            continue;
        }
        let sxy = sum_xy - sum_x * sum_y / sum_w;
        let syy = sum_yy - sum_y * sum_y / sum_w;
        let beta_value = sxy / sxx;
        let alpha_value = sum_y / sum_w - beta_value * sum_x / sum_w;
        let residual_var = ((syy - beta_value * sxy) / sum_w).max(0.0);
        alpha[idx] = Some(alpha_value);
        beta[idx] = Some(beta_value);
        residual_sigma[idx] = Some(residual_var.sqrt());
    }

    (alpha, beta, residual_sigma)
}

#[cfg(test)]
mod tests {
    use super::{ts_ew_regression_alpha_beta_residual_sigma, ts_ew_regression_beta_residual_sigma};

    #[test]
    fn ew_regression_recovers_linear_beta_and_zero_residual() {
        let x = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let y = vec![Some(3.0), Some(5.0), Some(7.0), Some(9.0)];
        let (beta, sigma) = ts_ew_regression_beta_residual_sigma(&y, &x, 4, 4, 2.0);

        assert!((beta[3].unwrap() - 2.0).abs() < 1e-10);
        assert!(sigma[3].unwrap().abs() < 1e-10);
    }

    #[test]
    fn ew_regression_recovers_intercept() {
        let x = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let y = vec![Some(3.0), Some(5.0), Some(7.0), Some(9.0)];
        let (alpha, beta, sigma) = ts_ew_regression_alpha_beta_residual_sigma(&y, &x, 4, 4, 2.0);

        assert!((alpha[3].unwrap() - 1.0).abs() < 1e-10);
        assert!((beta[3].unwrap() - 2.0).abs() < 1e-10);
        assert!(sigma[3].unwrap().abs() < 1e-10);
    }
}

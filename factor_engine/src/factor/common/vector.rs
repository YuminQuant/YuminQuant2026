pub fn map_unary<F>(values: &[Option<f64>], f: F) -> Vec<Option<f64>>
where
    F: Fn(f64) -> Option<f64>,
{
    values
        .iter()
        .map(|value| value.and_then(&f))
        .collect::<Vec<_>>()
}

pub fn map_binary<F>(left: &[Option<f64>], right: &[Option<f64>], f: F) -> Vec<Option<f64>>
where
    F: Fn(f64, f64) -> Option<f64>,
{
    left.iter()
        .zip(right)
        .map(|(left, right)| match (left, right) {
            (Some(left), Some(right)) => f(*left, *right),
            _ => None,
        })
        .collect::<Vec<_>>()
}

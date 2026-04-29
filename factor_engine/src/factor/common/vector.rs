pub fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

pub fn map_unary<F>(values: &[Option<f64>], f: F) -> Vec<Option<f64>>
where
    F: Fn(f64) -> Option<f64>,
{
    values
        .iter()
        .map(|value| clean(*value).and_then(&f))
        .collect::<Vec<_>>()
}

pub fn map_binary<F>(left: &[Option<f64>], right: &[Option<f64>], f: F) -> Vec<Option<f64>>
where
    F: Fn(f64, f64) -> Option<f64>,
{
    left.iter()
        .zip(right)
        .map(|(left, right)| match (clean(*left), clean(*right)) {
            (Some(left), Some(right)) => f(left, right),
            _ => None,
        })
        .collect::<Vec<_>>()
}

pub fn map_ternary<F>(
    first: &[Option<f64>],
    second: &[Option<f64>],
    third: &[Option<f64>],
    f: F,
) -> Vec<Option<f64>>
where
    F: Fn(f64, f64, f64) -> Option<f64>,
{
    first
        .iter()
        .zip(second)
        .zip(third)
        .map(
            |((first, second), third)| match (clean(*first), clean(*second), clean(*third)) {
                (Some(first), Some(second), Some(third)) => f(first, second, third),
                _ => None,
            },
        )
        .collect::<Vec<_>>()
}

pub fn map_quaternary<F>(
    first: &[Option<f64>],
    second: &[Option<f64>],
    third: &[Option<f64>],
    fourth: &[Option<f64>],
    f: F,
) -> Vec<Option<f64>>
where
    F: Fn(f64, f64, f64, f64) -> Option<f64>,
{
    first
        .iter()
        .zip(second)
        .zip(third)
        .zip(fourth)
        .map(|(((first, second), third), fourth)| {
            match (clean(*first), clean(*second), clean(*third), clean(*fourth)) {
                (Some(first), Some(second), Some(third), Some(fourth)) => {
                    f(first, second, third, fourth)
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>()
}

pub fn daily_vwap_from_amount_vol(amount: Option<f64>, vol: Option<f64>) -> Option<f64> {
    let (Some(amount), Some(vol)) = (clean(amount), clean(vol)) else {
        return None;
    };
    if vol.abs() <= f64::EPSILON {
        return None;
    }
    Some(amount * 10.0 / vol)
}

pub fn minute_vwap_from_amount_vol(amount: Option<f64>, vol: Option<f64>) -> Option<f64> {
    let (Some(amount), Some(vol)) = (clean(amount), clean(vol)) else {
        return None;
    };
    if vol.abs() <= f64::EPSILON {
        return None;
    }
    Some(amount / vol)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

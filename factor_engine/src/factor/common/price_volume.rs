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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_vwap_converts_thousand_yuan_and_lots_to_price() {
        assert_eq!(
            daily_vwap_from_amount_vol(Some(1000.0), Some(100.0)),
            Some(100.0)
        );
    }

    #[test]
    fn minute_vwap_uses_yuan_and_shares_directly() {
        assert_eq!(
            minute_vwap_from_amount_vol(Some(10000.0), Some(100.0)),
            Some(100.0)
        );
    }

    #[test]
    fn vwap_helpers_skip_missing_nan_and_zero_volume() {
        assert_eq!(daily_vwap_from_amount_vol(Some(1000.0), Some(0.0)), None);
        assert_eq!(
            minute_vwap_from_amount_vol(Some(f64::NAN), Some(100.0)),
            None
        );
        assert_eq!(minute_vwap_from_amount_vol(Some(10000.0), None), None);
    }
}

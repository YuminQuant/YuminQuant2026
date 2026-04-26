pub fn ts_delay(values: &[Option<f64>], periods: usize) -> Vec<Option<f64>> {
    if periods == 0 {
        return values.to_vec();
    }

    let mut output = vec![None; values.len()];
    for idx in periods..values.len() {
        output[idx] = values[idx - periods];
    }
    output
}

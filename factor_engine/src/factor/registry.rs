include!(concat!(env!("OUT_DIR"), "/factor_registry.rs"));

#[cfg(test)]
mod tests {
    use super::all_factors;
    use crate::core::DatasetId;

    #[test]
    fn analyst_factor_tags_follow_declared_data_dependencies() {
        let mut analyst_count = 0;
        for factor in all_factors() {
            for spec in factor.provided_specs() {
                let analyst = spec.dependencies.iter().any(|request| {
                    matches!(
                        request.dataset,
                        DatasetId::StockConsensus | DatasetId::StockAnalystReport
                    )
                });
                if !analyst {
                    continue;
                }
                analyst_count += 1;
                assert!(
                    spec.tags.iter().any(|tag| tag == "analyst"),
                    "{} lacks analyst tag",
                    spec.id
                );
                let financial = spec.dependencies.iter().any(|request| {
                    matches!(
                        request.dataset,
                        DatasetId::StockIncome
                            | DatasetId::StockBalanceSheet
                            | DatasetId::StockCashFlow
                    )
                });
                if financial {
                    assert!(
                        spec.tags.iter().any(|tag| tag == "fundamental"),
                        "{} lacks fundamental tag",
                        spec.id
                    );
                }
            }
        }
        assert!(analyst_count > 0);
    }
}

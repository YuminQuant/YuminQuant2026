include!(concat!(env!("OUT_DIR"), "/factor_registry.rs"));

#[cfg(test)]
mod tests {
    use super::all_factors;
    use crate::core::DatasetId;

    #[test]
    fn retired_analyst_and_fund_factors_remain_deprecated() {
        let retired = [
            "foa",
            "fom",
            "cov",
            "anncov",
            "firstcov",
            "pafr",
            "pafr_zscore",
            "fom_123mean",
            "rollingfyroe_resid_12mean",
            "rollingepfy_12mean",
            "rollingepfy_12mean_diff3",
            "peg_diff6",
        ];
        let specs: std::collections::HashMap<_, _> = all_factors()
            .into_iter()
            .flat_map(|factor| factor.provided_specs())
            .map(|spec| (spec.id.clone(), spec))
            .collect();
        for id in retired {
            let spec = specs
                .get(id)
                .unwrap_or_else(|| panic!("missing retired factor {id}"));
            assert!(
                spec.tags.iter().any(|tag| tag == "deprecated"),
                "{id} lacks deprecated tag"
            );
        }
    }

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

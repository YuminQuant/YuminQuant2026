use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrRtnMeanByT,
    "corr_rtn_mean_by_t",
    "corrRtnMean_byT",
    "corrRtnMean_byT",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::TimeRtn,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

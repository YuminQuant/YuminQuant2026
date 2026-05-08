use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrRtnStdByT,
    "corr_rtn_std_by_t",
    "corrRtnStd_byT",
    "corrRtnStd_byT",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::TimeRtn,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

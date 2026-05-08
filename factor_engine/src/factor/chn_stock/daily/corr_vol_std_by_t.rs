use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrVolStdByT,
    "corr_vol_std_by_t",
    "corrVolStd_byT",
    "corrVolStd_byT",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::TimeVol,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrRtnStdByV,
    "corr_rtn_std_by_v",
    "corrRtnStd_byV",
    "corrRtnStd_byV",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::VolumeRtn,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

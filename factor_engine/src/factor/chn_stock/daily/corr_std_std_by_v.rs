use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrStdStdByV,
    "corr_std_std_by_v",
    "corrStdStd_byV",
    "corrStdStd_byV",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::VolumeStd,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

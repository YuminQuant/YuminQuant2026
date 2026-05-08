use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrStdMeanByV,
    "corr_std_mean_by_v",
    "corrStdMean_byV",
    "corrStdMean_byV",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::VolumeStd,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

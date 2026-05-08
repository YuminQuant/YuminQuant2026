use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrRtnMeanByV,
    "corr_rtn_mean_by_v",
    "corrRtnMean_byV",
    "corrRtnMean_byV",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::VolumeRtn,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

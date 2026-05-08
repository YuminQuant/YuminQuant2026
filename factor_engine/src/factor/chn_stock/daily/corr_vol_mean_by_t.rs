use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrVolMeanByT,
    "corr_vol_mean_by_t",
    "corrVolMean_byT",
    "corrVolMean_byT",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::TimeVol,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

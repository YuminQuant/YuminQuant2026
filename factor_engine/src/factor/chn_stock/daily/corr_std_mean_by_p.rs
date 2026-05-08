use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrStdMeanByP,
    "corr_std_mean_by_p",
    "corrStdMean_byP",
    "corrStdMean_byP",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::PriceStd,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

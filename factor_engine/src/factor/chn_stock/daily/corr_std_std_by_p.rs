use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrStdStdByP,
    "corr_std_std_by_p",
    "corrStdStd_byP",
    "corrStdStd_byP",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::PriceStd,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

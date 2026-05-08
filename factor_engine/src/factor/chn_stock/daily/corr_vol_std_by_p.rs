use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrVolStdByP,
    "corr_vol_std_by_p",
    "corrVolStd_byP",
    "corrVolStd_byP",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::PriceVol,
        statistic: XyzqDomainCorrStatistic::Std
    }
);

use crate::factor::common::xyzq_domain_structure::{
    XyzqDomainCorrStatistic, XyzqDomainFactorKind, XyzqDomainFeature,
};

crate::define_xyzq_domain_factor!(
    StockDailyCorrVolMeanByP,
    "corr_vol_mean_by_p",
    "corrVolMean_byP",
    "corrVolMean_byP",
    XyzqDomainFactorKind::Corr {
        feature: XyzqDomainFeature::PriceVol,
        statistic: XyzqDomainCorrStatistic::Mean
    }
);

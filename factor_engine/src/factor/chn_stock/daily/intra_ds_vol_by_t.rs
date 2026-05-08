use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDsVolByT,
    "intra_ds_vol_by_t",
    "intraDSVol_byT",
    "intraDSVol_byT",
    XyzqDomainFactorKind::IntraDs {
        feature: XyzqDomainFeature::TimeVol
    }
);

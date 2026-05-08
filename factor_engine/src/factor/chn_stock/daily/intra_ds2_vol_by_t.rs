use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2VolByT,
    "intra_ds2_vol_by_t",
    "intraDS2Vol_byT",
    "intraDS2Vol_byT",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::TimeVol
    }
);

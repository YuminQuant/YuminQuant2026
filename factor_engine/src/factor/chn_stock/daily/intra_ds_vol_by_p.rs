use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDsVolByP,
    "intra_ds_vol_by_p",
    "intraDSVol_byP",
    "intraDSVol_byP",
    XyzqDomainFactorKind::IntraDs {
        feature: XyzqDomainFeature::PriceVol
    }
);

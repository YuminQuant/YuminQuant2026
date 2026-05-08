use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2VolByP,
    "intra_ds2_vol_by_p",
    "intraDS2Vol_byP",
    "intraDS2Vol_byP",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::PriceVol
    }
);

use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2StdByP,
    "intra_ds2_std_by_p",
    "intraDS2Std_byP",
    "intraDS2Std_byP",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::PriceStd
    }
);

use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2StdByV,
    "intra_ds2_std_by_v",
    "intraDS2Std_byV",
    "intraDS2Std_byV",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::VolumeStd
    }
);

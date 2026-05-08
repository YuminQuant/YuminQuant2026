use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDsStdByV,
    "intra_ds_std_by_v",
    "intraDSStd_byV",
    "intraDSStd_byV",
    XyzqDomainFactorKind::IntraDs {
        feature: XyzqDomainFeature::VolumeStd
    }
);

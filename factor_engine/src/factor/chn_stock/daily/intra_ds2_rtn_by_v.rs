use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2RtnByV,
    "intra_ds2_rtn_by_v",
    "intraDS2Rtn_byV",
    "intraDS2Rtn_byV",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::VolumeRtn
    }
);

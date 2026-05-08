use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainFeature};

crate::define_xyzq_domain_factor!(
    StockDailyIntraDs2RtnByT,
    "intra_ds2_rtn_by_t",
    "intraDS2Rtn_byT",
    "intraDS2Rtn_byT",
    XyzqDomainFactorKind::IntraDs2 {
        feature: XyzqDomainFeature::TimeRtn
    }
);

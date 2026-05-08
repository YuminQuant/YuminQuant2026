use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainKind};

crate::define_xyzq_domain_factor!(
    StockDailyPeerDsByV,
    "peer_ds_by_v",
    "peerDS_byV",
    "peerDS_byV",
    XyzqDomainFactorKind::PeerDs {
        domain: XyzqDomainKind::Volume
    }
);

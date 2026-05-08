use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainKind};

crate::define_xyzq_domain_factor!(
    StockDailyPeerDsByT,
    "peer_ds_by_t",
    "peerDS_byT",
    "peerDS_byT",
    XyzqDomainFactorKind::PeerDs {
        domain: XyzqDomainKind::Time
    }
);

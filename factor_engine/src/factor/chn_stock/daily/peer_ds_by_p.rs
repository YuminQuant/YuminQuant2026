use crate::factor::common::xyzq_domain_structure::{XyzqDomainFactorKind, XyzqDomainKind};

crate::define_xyzq_domain_factor!(
    StockDailyPeerDsByP,
    "peer_ds_by_p",
    "peerDS_byP",
    "peerDS_byP",
    XyzqDomainFactorKind::PeerDs {
        domain: XyzqDomainKind::Price
    }
);

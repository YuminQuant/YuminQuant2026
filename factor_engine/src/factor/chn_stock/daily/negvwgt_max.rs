use crate::factor::common::stock_daily_raw_ids::NEGVWGT_MAX_RAW_ID;
use crate::factor::common::xyzq_vshape_structure::{self, XyzqVshapeFactorKind};

crate::define_xyzq_vshape_structure_factor!(
    StockDailyNegvwgtMax,
    "negvwgt_max",
    "negVwgt_max",
    "negVwgt_max",
    XyzqVshapeFactorKind::RollingMean {
        raw_id: NEGVWGT_MAX_RAW_ID,
        window: xyzq_vshape_structure::default_window(),
    }
);

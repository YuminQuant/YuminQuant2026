use crate::factor::common::stock_daily_raw_ids::NEGV_MAX_RAW_ID;
use crate::factor::common::xyzq_vshape_structure::{self, XyzqVshapeFactorKind};

crate::define_xyzq_vshape_structure_factor!(
    StockDailyNegvMax,
    "negv_max",
    "negV_max",
    "negV_max",
    XyzqVshapeFactorKind::RollingMean {
        raw_id: NEGV_MAX_RAW_ID,
        window: xyzq_vshape_structure::default_window(),
    }
);

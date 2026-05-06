use crate::factor::common::stock_daily_raw_ids::NEGVWGT_MEAN_RAW_ID;
use crate::factor::common::xyzq_vshape_structure::{self, XyzqVshapeFactorKind};

crate::define_xyzq_vshape_structure_factor!(
    StockDailyNegvwgtMean,
    "negvwgt_mean",
    "negVwgt_mean",
    "negVwgt_mean",
    XyzqVshapeFactorKind::RollingMean {
        raw_id: NEGVWGT_MEAN_RAW_ID,
        window: xyzq_vshape_structure::default_window(),
    }
);

use crate::factor::common::stock_daily_raw_ids::GMM_MEANDIF_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyGmmMeandif,
    "gmm_meandif",
    "gmm_meandif",
    "gmm_meandif",
    GMM_MEANDIF_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

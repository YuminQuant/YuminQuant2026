use crate::factor::common::stock_daily_raw_ids::GMM_MEAN_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyGmmMean,
    "gmm_mean",
    "gmm_mean",
    "gmm_mean",
    GMM_MEAN_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

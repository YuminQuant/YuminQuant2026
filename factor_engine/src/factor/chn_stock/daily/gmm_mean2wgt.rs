use crate::factor::common::stock_daily_raw_ids::GMM_MEAN2WGT_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyGmmMean2wgt,
    "gmm_mean2wgt",
    "gmm_mean2wgt",
    "gmm_mean2wgt",
    GMM_MEAN2WGT_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::gmm_mean2wgt_smooth_window()
);

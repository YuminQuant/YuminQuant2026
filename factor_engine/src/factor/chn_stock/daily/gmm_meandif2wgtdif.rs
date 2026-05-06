use crate::factor::common::stock_daily_raw_ids::GMM_MEANDIF2WGTDIF_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyGmmMeandif2wgtdif,
    "gmm_meandif2wgtdif",
    "gmm_meandif2wgtdif",
    "gmm_meandif2wgtdif",
    GMM_MEANDIF2WGTDIF_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

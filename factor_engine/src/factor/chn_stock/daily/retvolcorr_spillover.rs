use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyRetvolcorrSpillover,
    "retvolcorr_spillover",
    "retvolcorr_spillover",
    "retvolcorr_spillover",
    XyzqIndustrySpilloverMode::RetVolCorr
);

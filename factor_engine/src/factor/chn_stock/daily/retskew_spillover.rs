use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyRetskewSpillover,
    "retskew_spillover",
    "retskew_spillover",
    "retskew_spillover",
    XyzqIndustrySpilloverMode::RetSkew
);

use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyTaylorretSpillover,
    "taylorret_spillover",
    "taylorret_spillover",
    "taylorret_spillover",
    XyzqIndustrySpilloverMode::TaylorRet
);

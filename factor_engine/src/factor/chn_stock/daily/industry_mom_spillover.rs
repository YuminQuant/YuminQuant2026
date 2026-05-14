use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyIndustryMomSpillover,
    "industry_mom_spillover",
    "industry_mom_spillover",
    "industry_mom_spillover",
    XyzqIndustrySpilloverMode::IndustryMomentum
);

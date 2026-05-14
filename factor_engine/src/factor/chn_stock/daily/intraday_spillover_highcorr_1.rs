use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyIntradaySpilloverHighcorr1,
    "intraday_spillover_highcorr_1",
    "intraday_spillover_highcorr_1",
    "intraday_spillover_highcorr_1",
    XyzqIndustrySpilloverMode::HighCorr1
);

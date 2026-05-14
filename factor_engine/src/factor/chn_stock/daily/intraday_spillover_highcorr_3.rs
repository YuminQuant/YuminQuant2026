use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyIntradaySpilloverHighcorr3,
    "intraday_spillover_highcorr_3",
    "intraday_spillover_highcorr_3",
    "intraday_spillover_highcorr_3",
    XyzqIndustrySpilloverMode::HighCorr3
);

use crate::define_xyzq_industry_spillover_factor;
use crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverMode;

define_xyzq_industry_spillover_factor!(
    StockDailyIntradaySpilloverHighcorr2,
    "intraday_spillover_highcorr_2",
    "intraday_spillover_highcorr_2",
    "intraday_spillover_highcorr_2",
    XyzqIndustrySpilloverMode::HighCorr2
);

use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::RTN5_MEAN_RAW_ID;
use crate::factor::common::xyzq_intraday_distribution::{self, XyzqFactorDef};
use crate::factor::Factor;

const DEF: XyzqFactorDef = XyzqFactorDef {
    id: "rtn5_mean",
    alias: "rtn5_mean",
    name: "rtn5_mean",
    raw_id: RTN5_MEAN_RAW_ID,
};

pub struct StockDailyRtn5Mean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRtn5Mean)
}

impl Factor for StockDailyRtn5Mean {
    fn spec(&self) -> FactorSpec {
        xyzq_intraday_distribution::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_intraday_distribution::raw_specs()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(
            xyzq_intraday_distribution::minute_compute_many(&raw_ids, context, data)?
                .into_iter()
                .next(),
        )
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_intraday_distribution::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_intraday_distribution::compute_factor(DEF, data)
    }
}

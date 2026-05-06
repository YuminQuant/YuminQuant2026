use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::REAL_VAR_RAW_ID;
use crate::factor::common::xyzq_intraday_distribution::{
    self, XyzqDistributionRawFamily, XyzqFactorDef,
};
use crate::factor::Factor;

const DEF: XyzqFactorDef = XyzqFactorDef {
    id: "real_var",
    alias: "real_var",
    name: "real_var",
    raw_id: REAL_VAR_RAW_ID,
};

pub struct StockDailyRealVar;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRealVar)
}

impl Factor for StockDailyRealVar {
    fn spec(&self) -> FactorSpec {
        xyzq_intraday_distribution::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_intraday_distribution::minute_return_distribution_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_minute_return_distribution_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_intraday_distribution::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqDistributionRawFamily::MinuteReturnDistribution,
        )?
        .into_iter()
        .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_intraday_distribution::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqDistributionRawFamily::MinuteReturnDistribution,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_intraday_distribution::compute_factor(DEF, data)
    }
}

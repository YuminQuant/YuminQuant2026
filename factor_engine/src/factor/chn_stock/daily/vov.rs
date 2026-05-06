use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::dbzq_5min_risk::{self, DbzqFactorDef, DbzqPostProcess};
use crate::factor::common::stock_daily_raw_ids::RV_5MIN_RAW_ID;
use crate::factor::Factor;

const DEF: DbzqFactorDef = DbzqFactorDef {
    id: "vov",
    alias: "VOV",
    name: "VOV",
    raw_id: RV_5MIN_RAW_ID,
    postprocess: DbzqPostProcess::Uncertainty,
};

pub struct StockDailyVov;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVov)
}

impl Factor for StockDailyVov {
    fn spec(&self) -> FactorSpec {
        dbzq_5min_risk::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        dbzq_5min_risk::raw_specs()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(
            dbzq_5min_risk::minute_compute_many(&raw_ids, context, data)?
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
        dbzq_5min_risk::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        dbzq_5min_risk::compute_factor(DEF, data)
    }
}

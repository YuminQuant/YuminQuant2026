use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::dbzq_5min_risk::{self, DbzqFactorDef, DbzqPostProcess, DbzqRawFamily};
use crate::factor::common::stock_daily_raw_ids::ID_RV_5MIN_RAW_ID;
use crate::factor::Factor;

const DEF: DbzqFactorDef = DbzqFactorDef {
    id: "id_rv_mean",
    alias: "ID_RV_mean",
    name: "ID RV Mean",
    raw_id: ID_RV_5MIN_RAW_ID,
    postprocess: DbzqPostProcess::WeekMean,
};

pub struct StockDailyIdRvMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIdRvMean)
}

impl Factor for StockDailyIdRvMean {
    fn spec(&self) -> FactorSpec {
        dbzq_5min_risk::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        dbzq_5min_risk::idiosyncratic_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "dbzq_id_5min_risk_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(dbzq_5min_risk::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            DbzqRawFamily::Idiosyncratic,
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
        dbzq_5min_risk::minute_compute_many_for(
            raw_ids,
            context,
            data,
            DbzqRawFamily::Idiosyncratic,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        dbzq_5min_risk::compute_factor(DEF, data)
    }
}

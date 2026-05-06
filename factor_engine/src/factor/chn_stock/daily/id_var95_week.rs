use crate::factor::common::stock_daily_raw_ids::ID_VAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVar95Week,
    "id_var95_week",
    "ID_VaR95_week",
    "ID VaR95 Week",
    ID_VAR95_5MIN_RAW_ID,
    WeekMean
);

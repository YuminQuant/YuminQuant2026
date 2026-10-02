//! Read-only loader benchmark against the local database.
use std::path::PathBuf;
use std::time::Instant;
use yq_factor_engine::config::EngineConfig;
use yq_factor_engine::data::{DataCatalog, DisclosureTableCache, MarketDataLoader};

fn main() -> yq_factor_engine::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let config = EngineConfig::discover(Some(PathBuf::from(&args[0]).join("config.toml")))?;
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root));
    let mut cache = DisclosureTableCache::default();
    let start = Instant::now();
    let mut rows = 0;
    let columns = ["org_name", "op_rt", "np", "eps", "rating", "min_price"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    for _ in 0..4 {
        let analyst = loader
            .load_stock_analyst_report_between_cached(&columns, 20250101, 20260424, &mut cache)?;
        rows += analyst.len;
        std::hint::black_box(&analyst);
        drop(analyst);
        let mainbz =
            loader.load_stock_main_business_cached(&[], 20260105, 20260424, 8, &mut cache)?;
        rows += mainbz.len;
        std::hint::black_box(&mainbz);
    }
    println!(
        "BENCH elapsed_ms={} rows={rows}",
        start.elapsed().as_millis()
    );
    Ok(())
}

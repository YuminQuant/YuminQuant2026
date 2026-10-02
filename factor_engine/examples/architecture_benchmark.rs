//! Isolated benchmark: reads the project database, writes only to an explicit scratch root.
use std::path::PathBuf;
use std::time::Instant;
use yq_factor_engine::config::EngineConfig;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::{Engine, RunRequest};

fn main() -> yq_factor_engine::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "project scratch start_date end_date");
    let project = PathBuf::from(&args[0]).canonicalize()?;
    let scratch = PathBuf::from(&args[1]).canonicalize()?;
    assert!(scratch.starts_with(project.join(".perf_arch_review")));
    let mut config = EngineConfig::discover(Some(project.join("config.toml")))?;
    config.factor_root = scratch;
    let engine = Engine::new(config);
    engine.write_metadata()?;
    let start = Instant::now();
    let report = engine.run(&RunRequest {
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        start_date: args[2].parse().unwrap(),
        end_date: args[3].parse().unwrap(),
        factor_ids: Some(
            [
                "roe_enhance",
                "abcfo",
                "sfli2",
                "nol2",
                "ep_sq_gauss_resid",
                "sp_sq_gauss_resid",
                "ret20",
                "ret20_adj",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        ),
        tags: None,
        config_path: Some(project.join("config.toml")),
        dry_run: false,
        factor_batch_size: 5,
        date_batch_size: 60,
        threads: Some(4),
        profile: true,
        refresh_minute_cache: false,
    })?;
    let load: u128 = report.profiles.iter().map(|p| p.load_ms).sum();
    let compute: u128 = report.profiles.iter().map(|p| p.compute_ms).sum();
    let write: u128 = report.profiles.iter().map(|p| p.write_ms).sum();
    println!("BENCH elapsed_ms={} load_ms={load} compute_ms={compute} write_ms={write} days={} factors={}", start.elapsed().as_millis(), report.target_dates.len(), report.factor_count);
    Ok(())
}

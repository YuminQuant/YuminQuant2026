use std::collections::HashMap;
use std::path::PathBuf;

use yq_factor_engine::config::EngineConfig;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::{Engine, Result, RunRequest};

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run_cli() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args[0] == "help" || args[0] == "--help" {
        print_help();
        return Ok(());
    }

    match args[0].as_str() {
        "metadata" => {
            let flags = parse_flags(&args[1..])?;
            let engine = engine_from_flags(&flags)?;
            let count = engine.write_metadata()?;
            println!("metadata complete");
            println!("factors: {count}");
        }
        "list" => {
            let flags = parse_flags(&args[1..])?;
            let engine = engine_from_flags(&flags)?;
            let asset_filter = flags
                .get("asset")
                .and_then(|value| AssetClass::parse(value));
            let frequency_filter = flags
                .get("frequency")
                .and_then(|value| Frequency::parse(value));
            let ids_only = flags
                .get("ids-only")
                .map(|value| value == "true" || value == "1")
                .unwrap_or(false);
            for row in engine.read_metadata()? {
                if asset_filter.is_some_and(|asset| row.asset_class != asset.as_str()) {
                    continue;
                }
                if frequency_filter.is_some_and(|frequency| row.frequency != frequency.as_str()) {
                    continue;
                }
                if ids_only {
                    println!("{}", row.factor_id);
                } else {
                    println!(
                        "{} | asset={} | frequency={} | version={} | tags={}",
                        row.factor_id,
                        row.asset_class,
                        row.frequency,
                        row.version,
                        row.tags.join(",")
                    );
                }
            }
        }
        "plan" => {
            let request = parse_run_request(&args[1..], true)?;
            let engine = Engine::from_request(&request)?;
            let report = engine.plan(&request)?;
            print_report("plan", &report);
        }
        "run" => {
            let request = parse_run_request(&args[1..], false)?;
            let engine = Engine::from_request(&request)?;
            let report = engine.run(&request)?;
            print_report("run", &report);
        }
        command => {
            return Err(yq_factor_engine::error::err(format!(
                "unknown command: {command}"
            )));
        }
    }
    Ok(())
}

fn parse_run_request(args: &[String], dry_run: bool) -> Result<RunRequest> {
    let flags = parse_flags(args)?;
    let asset_class = flags
        .get("asset")
        .and_then(|value| AssetClass::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --asset stock|future"))?;
    let frequency = flags
        .get("frequency")
        .and_then(|value| Frequency::parse(value))
        .ok_or_else(|| {
            yq_factor_engine::error::err("missing or invalid --frequency daily|minute_1m")
        })?;
    let start_date = flags
        .get("start-date")
        .ok_or_else(|| yq_factor_engine::error::err("missing --start-date YYYYMMDD"))?
        .parse::<i32>()?;
    let end_date = flags
        .get("end-date")
        .ok_or_else(|| yq_factor_engine::error::err("missing --end-date YYYYMMDD"))?
        .parse::<i32>()?;
    let factor_ids = flags.get("factors").map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    });
    let tags = flags.get("tags").map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    });
    if factor_ids.is_some() && tags.is_some() {
        return Err(yq_factor_engine::error::err(
            "--factors and --tags cannot be used together",
        ));
    }
    let config_path = flags.get("config").map(PathBuf::from);
    Ok(RunRequest {
        asset_class,
        frequency,
        start_date,
        end_date,
        factor_ids,
        tags,
        config_path,
        dry_run,
    })
}

fn engine_from_flags(flags: &HashMap<String, String>) -> Result<Engine> {
    let config_path = flags.get("config").map(PathBuf::from);
    Ok(Engine::new(EngineConfig::discover(config_path)?))
}

fn parse_flags(args: &[String]) -> Result<HashMap<String, String>> {
    let mut flags = HashMap::new();
    let mut idx = 0;
    while idx < args.len() {
        let key = args[idx].strip_prefix("--").ok_or_else(|| {
            yq_factor_engine::error::err(format!("expected --flag, got {}", args[idx]))
        })?;
        let value = args
            .get(idx + 1)
            .ok_or_else(|| yq_factor_engine::error::err(format!("missing value for --{key}")))?;
        flags.insert(key.to_string(), value.clone());
        idx += 2;
    }
    Ok(flags)
}

fn print_report(label: &str, report: &yq_factor_engine::RunReport) {
    if let Some(message) = &report.status_message {
        println!("{message}");
        return;
    }
    println!("{} complete", label);
    println!("factors: {}", report.factor_count);
    println!("output files: {}", report.output_file_count);
    println!("load_start_date: {}", report.load_start_date);
    println!("target dates: {}", report.target_dates.len());
    println!("selected factors:");
    for factor_id in &report.selected_factor_ids {
        println!("  {}", factor_id);
    }
    for request in &report.loaded_requests {
        println!(
            "load {} columns={}",
            request.dataset.as_str(),
            request.columns.join(",")
        );
    }
}

fn print_help() {
    println!("YuminQuant factor engine MVP");
    println!();
    println!("commands:");
    println!("  metadata [--config D:/path/to/config.toml]");
    println!("  list [--asset stock|future] [--frequency daily|minute_1m] [--ids-only true]");
    println!("  plan --asset stock|future --frequency daily|minute_1m --start-date YYYYMMDD --end-date YYYYMMDD");
    println!("  run  --asset stock|future --frequency daily|minute_1m --start-date YYYYMMDD --end-date YYYYMMDD");
    println!();
    println!("optional flags:");
    println!("  --factors factor_id[,factor_id...]");
    println!("  --tags tag[,tag...]");
    println!("  --config D:/path/to/config.toml");
}

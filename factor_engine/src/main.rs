use std::collections::HashMap;
use std::path::PathBuf;

use yq_factor_engine::config::EngineConfig;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::engine::DEFAULT_FACTOR_BATCH_SIZE;
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
    let start_date = parse_yyyymmdd(
        flags
            .get("start-date")
            .ok_or_else(|| yq_factor_engine::error::err("missing --start-date YYYYMMDD"))?,
        "start-date",
    )?;
    let end_date = parse_yyyymmdd(
        flags
            .get("end-date")
            .ok_or_else(|| yq_factor_engine::error::err("missing --end-date YYYYMMDD"))?,
        "end-date",
    )?;
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
    let factor_batch_size = match flags.get("factor-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--factor-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_FACTOR_BATCH_SIZE,
    };
    let threads = match flags.get("threads") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--threads must be greater than 0",
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let config_path = flags.get("config").map(PathBuf::from);
    let profile = flag_enabled(&flags, "profile");
    Ok(RunRequest {
        asset_class,
        frequency,
        start_date,
        end_date,
        factor_ids,
        tags,
        config_path,
        dry_run,
        factor_batch_size,
        threads,
        profile,
    })
}

fn parse_yyyymmdd(value: &str, name: &str) -> Result<i32> {
    if value.len() != 8 || !value.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(yq_factor_engine::error::err(format!(
            "--{name} must be an 8-digit YYYYMMDD date, got {value}"
        )));
    }
    Ok(value.parse::<i32>()?)
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
        if args
            .get(idx + 1)
            .is_some_and(|value| !value.starts_with("--"))
        {
            flags.insert(key.to_string(), args[idx + 1].clone());
            idx += 2;
        } else {
            flags.insert(key.to_string(), "true".to_string());
            idx += 1;
        }
    }
    Ok(flags)
}

fn flag_enabled(flags: &HashMap<String, String>, name: &str) -> bool {
    flags
        .get(name)
        .map(|value| value == "true" || value == "1" || value.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
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
    if let (Some(start_date), Some(end_date)) =
        (report.effective_start_date, report.effective_end_date)
    {
        println!("effective date range: {}..{}", start_date, end_date);
    }
    println!("target dates: {}", report.target_dates.len());
    println!("date batches: {}", report.date_batch_count);
    println!("factor batches: {}", report.factor_batch_count);
    println!("execution batches: {}", report.execution_batch_count);
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
    if !report.profiles.is_empty() {
        println!("profile:");
        for batch in &report.profiles {
            println!(
                "  date_batch={} factor_batch={} dates={}..{} factors={} load_ms={} compute_ms={} write_ms={}",
                batch.date_batch_index,
                batch.factor_batch_index,
                batch.start_date,
                batch.end_date,
                batch.factor_count,
                batch.load_ms,
                batch.compute_ms,
                batch.write_ms
            );
            for factor in &batch.factors {
                println!(
                    "    {} rows={} non_null={}",
                    factor.factor_id, factor.row_count, factor.non_null_count
                );
            }
        }
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
    println!("  --factor-batch-size N (default 64)");
    println!("  --threads N");
    println!("  --profile");
    println!("  --config D:/path/to/config.toml");
}

#[cfg(test)]
mod tests {
    use super::{flag_enabled, parse_flags, parse_yyyymmdd};

    #[test]
    fn parse_yyyymmdd_rejects_non_eight_digit_input() {
        assert_eq!(parse_yyyymmdd("20260424", "end-date").unwrap(), 20260424);
        assert!(parse_yyyymmdd("2026424", "end-date").is_err());
        assert!(parse_yyyymmdd("2026-04-24", "end-date").is_err());
    }

    #[test]
    fn parse_flags_supports_boolean_flags() {
        let flags = parse_flags(&[
            "--profile".to_string(),
            "--asset".to_string(),
            "stock".to_string(),
        ])
        .expect("flags");

        assert!(flag_enabled(&flags, "profile"));
        assert_eq!(flags.get("asset").map(String::as_str), Some("stock"));
    }
}

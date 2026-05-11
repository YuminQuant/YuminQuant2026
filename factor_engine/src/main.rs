use std::collections::HashMap;
use std::path::PathBuf;

use yq_factor_engine::backtest::request::{
    BacktestRunRequest, FactorFill, LimitSide, NeutralizeSpec, RebalanceRule,
    DEFAULT_BACKTEST_LABEL, DEFAULT_BENCHMARK,
    DEFAULT_DATE_BATCH_SIZE as DEFAULT_BACKTEST_DATE_BATCH_SIZE, DEFAULT_EXCLUDE_LIMIT,
    DEFAULT_EXCLUDE_ST, DEFAULT_FACTOR_BATCH_SIZE as DEFAULT_BACKTEST_FACTOR_BATCH_SIZE,
    DEFAULT_GROUPS, DEFAULT_UNIVERSE,
};
use yq_factor_engine::barra::engine::DEFAULT_BARRA_MODEL;
use yq_factor_engine::config::EngineConfig;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::engine::{DEFAULT_DATE_BATCH_SIZE, DEFAULT_FACTOR_BATCH_SIZE};
use yq_factor_engine::{
    BacktestEngine, BacktestRunReport, BarraEngine, BarraRunRequest, Engine, LabelEngine,
    LabelRunRequest, Result, RunRequest,
};

const DEFAULT_LABEL_BATCH_SIZE: usize = 5;

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
            let tags_filter = flags.get("tags").map(|value| parse_csv_values(value));
            for row in engine.read_metadata()? {
                if asset_filter.is_some_and(|asset| row.asset_class != asset.as_str()) {
                    continue;
                }
                if frequency_filter.is_some_and(|frequency| row.frequency != frequency.as_str()) {
                    continue;
                }
                if !matches_tag_filter(&row.tags, &tags_filter) {
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
        "label-metadata" => {
            let flags = parse_flags(&args[1..])?;
            let engine = label_engine_from_flags(&flags)?;
            let count = engine.write_metadata()?;
            println!("label metadata complete");
            println!("labels: {count}");
        }
        "label-list" => {
            let flags = parse_flags(&args[1..])?;
            let engine = label_engine_from_flags(&flags)?;
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
            let tags_filter = flags.get("tags").map(|value| parse_csv_values(value));
            for row in engine.read_metadata()? {
                if asset_filter.is_some_and(|asset| row.asset_class != asset.as_str()) {
                    continue;
                }
                if frequency_filter.is_some_and(|frequency| row.frequency != frequency.as_str()) {
                    continue;
                }
                if !matches_tag_filter(&row.tags, &tags_filter) {
                    continue;
                }
                if ids_only {
                    println!("{}", row.label_id);
                } else {
                    println!(
                        "{} | asset={} | frequency={} | version={} | tags={}",
                        row.label_id,
                        row.asset_class,
                        row.frequency,
                        row.version,
                        row.tags.join(",")
                    );
                }
            }
        }
        "barra-metadata" => {
            let flags = parse_flags(&args[1..])?;
            let engine = barra_engine_from_flags(&flags)?;
            let count = engine.write_metadata()?;
            println!("barra metadata complete");
            println!("exposures: {count}");
        }
        "barra-list" => {
            let flags = parse_flags(&args[1..])?;
            let engine = barra_engine_from_flags(&flags)?;
            let asset_filter = flags
                .get("asset")
                .and_then(|value| AssetClass::parse(value));
            let frequency_filter = flags
                .get("frequency")
                .and_then(|value| Frequency::parse(value));
            let model_filter = flags
                .get("model")
                .cloned()
                .unwrap_or_else(|| DEFAULT_BARRA_MODEL.to_string());
            if !model_filter.eq_ignore_ascii_case(DEFAULT_BARRA_MODEL) {
                return Err(yq_factor_engine::error::err(format!(
                    "--model currently only supports {}",
                    DEFAULT_BARRA_MODEL
                )));
            }
            let ids_only = flags
                .get("ids-only")
                .map(|value| value == "true" || value == "1")
                .unwrap_or(false);
            let tags_filter = flags.get("tags").map(|value| parse_csv_values(value));
            for row in engine.read_metadata()? {
                if !row.model.eq_ignore_ascii_case(&model_filter) {
                    continue;
                }
                if asset_filter.is_some_and(|asset| row.asset_class != asset.as_str()) {
                    continue;
                }
                if frequency_filter.is_some_and(|frequency| row.frequency != frequency.as_str()) {
                    continue;
                }
                if !matches_tag_filter(&row.tags, &tags_filter) {
                    continue;
                }
                if ids_only {
                    println!("{}", row.exposure_id);
                } else {
                    println!(
                        "{} | model={} | asset={} | frequency={} | version={} | tags={}",
                        row.exposure_id,
                        row.model,
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
        "label-plan" => {
            let request = parse_label_run_request(&args[1..], true)?;
            let engine = LabelEngine::from_request(&request)?;
            let report = engine.plan(&request)?;
            print_label_report("label-plan", &report);
        }
        "label-run" => {
            let request = parse_label_run_request(&args[1..], false)?;
            let engine = LabelEngine::from_request(&request)?;
            let report = engine.run(&request)?;
            print_label_report("label-run", &report);
        }
        "barra-plan" => {
            let request = parse_barra_run_request(&args[1..], true)?;
            let engine = BarraEngine::from_request(&request)?;
            let report = engine.plan(&request)?;
            print_barra_report("barra-plan", &report);
        }
        "barra-run" => {
            let request = parse_barra_run_request(&args[1..], false)?;
            let engine = BarraEngine::from_request(&request)?;
            let report = engine.run(&request)?;
            print_barra_report("barra-run", &report);
        }
        "backtest" => {
            let request = parse_backtest_run_request(&args[1..])?;
            let engine = BacktestEngine::from_request(&request)?;
            let report = engine.run(&request)?;
            print_backtest_report(&report);
        }
        command => {
            return Err(yq_factor_engine::error::err(format!(
                "unknown command: {command}"
            )));
        }
    }
    Ok(())
}

fn parse_barra_run_request(args: &[String], dry_run: bool) -> Result<BarraRunRequest> {
    let flags = parse_flags(args)?;
    let asset_class = flags
        .get("asset")
        .and_then(|value| AssetClass::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --asset stock|future"))?;
    let frequency = flags
        .get("frequency")
        .and_then(|value| Frequency::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --frequency daily"))?;
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
    let exposure_ids = flags.get("exposures").map(|value| parse_csv_values(value));
    let tags = flags.get("tags").map(|value| parse_csv_values(value));
    let families = flags.get("families").map(|value| parse_csv_values(value));
    if exposure_ids.is_some() && tags.is_some() {
        return Err(yq_factor_engine::error::err(
            "--exposures and --tags cannot be used together",
        ));
    }
    let exposure_batch_size = match flags.get("exposure-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--exposure-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => 1,
    };
    let date_batch_size = match flags.get("date-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--date-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_DATE_BATCH_SIZE,
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
    let raw_model = flags
        .get("model")
        .cloned()
        .unwrap_or_else(|| DEFAULT_BARRA_MODEL.to_string());
    if !raw_model.eq_ignore_ascii_case(DEFAULT_BARRA_MODEL) {
        return Err(yq_factor_engine::error::err(format!(
            "--model currently only supports {}",
            DEFAULT_BARRA_MODEL
        )));
    }
    let model = DEFAULT_BARRA_MODEL.to_string();
    Ok(BarraRunRequest {
        asset_class,
        frequency,
        model,
        start_date,
        end_date,
        exposure_ids,
        tags,
        families,
        config_path,
        dry_run,
        exposure_batch_size,
        date_batch_size,
        threads,
        profile,
    })
}

fn parse_backtest_run_request(args: &[String]) -> Result<BacktestRunRequest> {
    let flags = parse_flags(args)?;
    if flags.contains_key("write-detail") {
        return Err(yq_factor_engine::error::err(
            "--write-detail has been removed; backtest now always writes returns, ic and factor_stats files",
        ));
    }
    let asset_class = flags
        .get("asset")
        .and_then(|value| AssetClass::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --asset stock"))?;
    let frequency = flags
        .get("frequency")
        .and_then(|value| Frequency::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --frequency daily"))?;
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
    let factor_ids = flags.get("factors").map(|value| parse_csv_values(value));
    let tags = flags.get("tags").map(|value| parse_csv_values(value));
    let all_factors = flag_enabled(&flags, "all-factors");
    let selection_count =
        usize::from(factor_ids.is_some()) + usize::from(tags.is_some()) + usize::from(all_factors);
    if selection_count > 1 {
        return Err(yq_factor_engine::error::err(
            "--factors, --tags and --all-factors cannot be used together",
        ));
    }
    if selection_count == 0 {
        return Err(yq_factor_engine::error::err(
            "backtest requires --factors, --tags or --all-factors",
        ));
    }
    let factor_root = flags.get("factor-root").map(PathBuf::from);
    let groups = match flags.get("groups") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--groups must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_GROUPS,
    };
    let rebalance = flags
        .get("rebalance")
        .map(|value| RebalanceRule::parse(value))
        .transpose()?
        .unwrap_or(RebalanceRule::Daily);
    let neutralize = flags
        .get("neutralize")
        .map(|value| NeutralizeSpec::parse(value))
        .transpose()?
        .unwrap_or(NeutralizeSpec::None);
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
        None => DEFAULT_BACKTEST_FACTOR_BATCH_SIZE,
    };
    let date_batch_size = match flags.get("date-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--date-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_BACKTEST_DATE_BATCH_SIZE,
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
    let factor_fill = flags
        .get("factor-fill")
        .map(|value| FactorFill::parse(value))
        .transpose()?
        .unwrap_or(FactorFill::None);
    Ok(BacktestRunRequest {
        asset_class,
        frequency,
        start_date,
        end_date,
        factor_ids,
        tags,
        all_factors,
        factor_root,
        label_id: flags
            .get("label")
            .cloned()
            .unwrap_or_else(|| DEFAULT_BACKTEST_LABEL.to_string()),
        groups,
        rebalance,
        neutralize,
        universe: flags
            .get("universe")
            .cloned()
            .unwrap_or_else(|| DEFAULT_UNIVERSE.to_string()),
        benchmark: flags
            .get("benchmark")
            .cloned()
            .unwrap_or_else(|| DEFAULT_BENCHMARK.to_string()),
        exclude_limit: flag_bool(&flags, "exclude-limit", DEFAULT_EXCLUDE_LIMIT),
        exclude_st: flag_bool(&flags, "exclude-st", DEFAULT_EXCLUDE_ST),
        limit_side: flags
            .get("limit-side")
            .map(|value| LimitSide::parse(value))
            .transpose()?
            .unwrap_or(LimitSide::Both),
        factor_batch_size,
        date_batch_size,
        threads,
        factor_fill,
        output_dir: flags.get("output-dir").map(PathBuf::from),
        config_path: flags.get("config").map(PathBuf::from),
    })
}

fn parse_label_run_request(args: &[String], dry_run: bool) -> Result<LabelRunRequest> {
    let flags = parse_flags(args)?;
    let asset_class = flags
        .get("asset")
        .and_then(|value| AssetClass::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --asset stock|future"))?;
    let frequency = flags
        .get("frequency")
        .and_then(|value| Frequency::parse(value))
        .ok_or_else(|| yq_factor_engine::error::err("missing or invalid --frequency daily"))?;
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
    let label_ids = flags.get("labels").map(|value| parse_csv_values(value));
    let tags = flags.get("tags").map(|value| parse_csv_values(value));
    if label_ids.is_some() && tags.is_some() {
        return Err(yq_factor_engine::error::err(
            "--labels and --tags cannot be used together",
        ));
    }
    let label_batch_size = parse_label_batch_size(&flags)?;
    let date_batch_size = match flags.get("date-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--date-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_DATE_BATCH_SIZE,
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
    let refresh_label_cache = flag_enabled(&flags, "refresh-label-cache");
    Ok(LabelRunRequest {
        asset_class,
        frequency,
        start_date,
        end_date,
        label_ids,
        tags,
        config_path,
        dry_run,
        label_batch_size,
        date_batch_size,
        threads,
        profile,
        refresh_label_cache,
    })
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
    let factor_ids = flags.get("factors").map(|value| parse_csv_values(value));
    let tags = flags.get("tags").map(|value| parse_csv_values(value));
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
    let date_batch_size = match flags.get("date-batch-size") {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--date-batch-size must be greater than 0",
                ));
            }
            parsed
        }
        None => DEFAULT_DATE_BATCH_SIZE,
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
    let refresh_minute_cache = flag_enabled(&flags, "refresh-minute-cache");
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
        date_batch_size,
        threads,
        profile,
        refresh_minute_cache,
    })
}

fn parse_label_batch_size(flags: &HashMap<String, String>) -> Result<usize> {
    let batch_size = flags.get("label-batch-size");
    let batch_num = flags.get("label-batch-num");
    if let (Some(batch_size), Some(batch_num)) = (batch_size, batch_num) {
        if batch_size != batch_num {
            return Err(yq_factor_engine::error::err(
                "--label-batch-size and --label-batch-num cannot be different",
            ));
        }
    }
    let value = batch_size.or(batch_num);
    match value {
        Some(value) => {
            let parsed = value.parse::<usize>()?;
            if parsed == 0 {
                return Err(yq_factor_engine::error::err(
                    "--label-batch-size/--label-batch-num must be greater than 0",
                ));
            }
            Ok(parsed)
        }
        None => Ok(DEFAULT_LABEL_BATCH_SIZE),
    }
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

fn label_engine_from_flags(flags: &HashMap<String, String>) -> Result<LabelEngine> {
    let config_path = flags.get("config").map(PathBuf::from);
    Ok(LabelEngine::new(EngineConfig::discover(config_path)?))
}

fn barra_engine_from_flags(flags: &HashMap<String, String>) -> Result<BarraEngine> {
    if let Some(model) = flags.get("model") {
        if !model.eq_ignore_ascii_case(DEFAULT_BARRA_MODEL) {
            return Err(yq_factor_engine::error::err(format!(
                "--model currently only supports {}",
                DEFAULT_BARRA_MODEL
            )));
        }
    }
    let config_path = flags.get("config").map(PathBuf::from);
    Ok(BarraEngine::new(EngineConfig::discover(config_path)?))
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

fn flag_bool(flags: &HashMap<String, String>, name: &str, default: bool) -> bool {
    flags
        .get(name)
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => true,
            "false" | "0" | "no" | "n" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn parse_csv_values(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn matches_tag_filter(row_tags: &[String], tags_filter: &Option<Vec<String>>) -> bool {
    let asks_for_deprecated = tags_filter
        .as_ref()
        .is_some_and(|tags| tags.iter().any(|tag| tag == "deprecated"));
    if !asks_for_deprecated && row_tags.iter().any(|tag| tag == "deprecated") {
        return false;
    }
    match tags_filter {
        Some(tags) => tags
            .iter()
            .all(|tag| row_tags.iter().any(|row_tag| row_tag == tag)),
        None => true,
    }
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
    if !report.execution_stages.is_empty() {
        println!("execution stages: {}", report.execution_stages.join(","));
    }
    println!("date batches: {}", report.date_batch_count);
    println!("factor batches: {}", report.factor_batch_count);
    println!("execution batches: {}", report.execution_batch_count);
    println!("selected factors:");
    for factor_id in &report.selected_factor_ids {
        println!("  {}", factor_id);
    }
    for request in &report.loaded_requests {
        let entity = request
            .entity_id
            .as_ref()
            .map(|value| format!(" entity={value}"))
            .unwrap_or_default();
        println!(
            "load {}{} columns={}",
            request.dataset.as_str(),
            entity,
            request.columns.join(",")
        );
    }
    for request in &report.loaded_intraday_raw_requests {
        println!(
            "load intraday_raw {} daily_lookback={}",
            request.raw_id, request.daily_lookback
        );
    }
    if !report.profiles.is_empty() {
        println!("profile:");
        for batch in &report.profiles {
            println!(
                "  stage={} date_batch={} factor_batch={} dates={}..{} factors={} load_ms={} compute_ms={} write_ms={}",
                batch.stage,
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

fn print_label_report(label: &str, report: &yq_factor_engine::LabelRunReport) {
    if let Some(message) = &report.status_message {
        println!("{message}");
        return;
    }
    println!("{} complete", label);
    println!("labels: {}", report.label_count);
    println!("output files: {}", report.output_file_count);
    println!("max_lookahead: {}", report.max_lookahead);
    if let (Some(start_date), Some(end_date)) =
        (report.effective_start_date, report.effective_end_date)
    {
        println!("effective date range: {}..{}", start_date, end_date);
    }
    println!("target dates: {}", report.target_dates.len());
    println!("skipped dates: {}", report.skipped_dates.len());
    println!("date batches: {}", report.date_batch_count);
    println!("label batches: {}", report.label_batch_count);
    println!("execution batches: {}", report.execution_batch_count);
    println!("selected labels:");
    for label_id in &report.selected_label_ids {
        println!("  {}", label_id);
    }
    for request in &report.loaded_requests {
        let entity = request
            .entity_id
            .as_ref()
            .map(|value| format!(" entity={value}"))
            .unwrap_or_default();
        println!(
            "load {}{} columns={}",
            request.dataset.as_str(),
            entity,
            request.columns.join(",")
        );
    }
    for request in &report.loaded_intraday_raw_requests {
        println!(
            "load label_intraday_raw {} daily_lookback={}",
            request.raw_id, request.daily_lookback
        );
    }
    if !report.profiles.is_empty() {
        println!("profile:");
        for batch in &report.profiles {
            println!(
                "  stage={} date_batch={} label_batch={} dates={}..{} labels={} load_ms={} compute_ms={} write_ms={}",
                batch.stage,
                batch.date_batch_index,
                batch.factor_batch_index,
                batch.start_date,
                batch.end_date,
                batch.factor_count,
                batch.load_ms,
                batch.compute_ms,
                batch.write_ms
            );
            for label in &batch.factors {
                println!(
                    "    {} rows={} non_null={}",
                    label.factor_id, label.row_count, label.non_null_count
                );
            }
        }
    }
}

fn print_barra_report(label: &str, report: &yq_factor_engine::BarraRunReport) {
    if let Some(message) = &report.status_message {
        println!("{message}");
        return;
    }
    println!("{} complete", label);
    println!("model: {}", report.model);
    println!("exposures: {}", report.exposure_count);
    println!("output files: {}", report.output_file_count);
    println!("load_start_date: {}", report.load_start_date);
    if let (Some(start_date), Some(end_date)) =
        (report.effective_start_date, report.effective_end_date)
    {
        println!("effective date range: {}..{}", start_date, end_date);
    }
    println!("target dates: {}", report.target_dates.len());
    println!("date batches: {}", report.date_batch_count);
    println!("exposure batches: {}", report.exposure_batch_count);
    println!("execution batches: {}", report.execution_batch_count);
    println!("selected exposures:");
    for exposure_id in &report.selected_exposure_ids {
        println!("  {}", exposure_id);
    }
    for request in &report.loaded_requests {
        let entity = request
            .entity_id
            .as_ref()
            .map(|value| format!(" entity={value}"))
            .unwrap_or_default();
        println!(
            "load {}{} columns={}",
            request.dataset.as_str(),
            entity,
            request.columns.join(",")
        );
    }
    if !report.profiles.is_empty() {
        println!("profile:");
        for batch in &report.profiles {
            println!(
                "  stage={} date_batch={} exposure_batch={} dates={}..{} exposures={} load_ms={} compute_ms={} write_ms={}",
                batch.stage,
                batch.date_batch_index,
                batch.factor_batch_index,
                batch.start_date,
                batch.end_date,
                batch.factor_count,
                batch.load_ms,
                batch.compute_ms,
                batch.write_ms
            );
            for exposure in &batch.factors {
                println!(
                    "    {} rows={} non_null={}",
                    exposure.factor_id, exposure.row_count, exposure.non_null_count
                );
            }
        }
    }
}

fn print_backtest_report(report: &BacktestRunReport) {
    println!("backtest complete");
    println!("factors: {}", report.factor_count);
    println!("rebalance dates: {}", report.rebalance_count);
    println!("output_dir: {}", report.output_dir.display());
    println!("output files: {}", report.output_files.len());
    for path in &report.output_files {
        println!("  {}", path.display());
    }
    println!("selected factors:");
    let display_limit = 50;
    for factor_id in report.selected_factor_ids.iter().take(display_limit) {
        println!("  {}", factor_id);
    }
    if report.selected_factor_ids.len() > display_limit {
        println!(
            "  ... {} more",
            report.selected_factor_ids.len() - display_limit
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
    println!("  label-metadata [--config D:/path/to/config.toml]");
    println!("  label-list [--asset stock|future] [--frequency daily] [--ids-only true]");
    println!(
        "  label-plan --asset stock --frequency daily --start-date YYYYMMDD --end-date YYYYMMDD"
    );
    println!(
        "  label-run  --asset stock --frequency daily --start-date YYYYMMDD --end-date YYYYMMDD"
    );
    println!("  barra-metadata [--config D:/path/to/config.toml]");
    println!("  barra-list [--asset stock|future] [--frequency daily] [--ids-only true]");
    println!(
        "  barra-plan --asset stock --frequency daily --start-date YYYYMMDD --end-date YYYYMMDD"
    );
    println!(
        "  barra-run  --asset stock --frequency daily --start-date YYYYMMDD --end-date YYYYMMDD"
    );
    println!(
        "  backtest --asset stock --frequency daily --start-date YYYYMMDD --end-date YYYYMMDD --factors factor_id[,factor_id...]"
    );
    println!();
    println!("optional flags:");
    println!("  --factors factor_id[,factor_id...]");
    println!("  --all-factors (backtest all non-deprecated factors)");
    println!("  --labels label_id[,label_id...]");
    println!("  --exposures exposure_id[,exposure_id...]");
    println!("  --families barra_family[,barra_family...]");
    println!("  --model CNE6");
    println!("  --tags tag[,tag...]");
    println!("  --label label_id (backtest default future_vwap_return_1d)");
    println!(
        "  --factor-root D:/path/to/root (backtest external factor root; supports --factors or --all-factors)"
    );
    println!("  --groups N (backtest default 10)");
    println!("  --rebalance daily|N|weekly|biweekly|monthly|quarterly");
    println!("  --neutralize none|sector|barra:SIZE|barra:SIZE+sector|barra:all+sector (all = primary Barra style exposures)");
    println!("  --universe mkt_all|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id");
    println!("  --benchmark mkt_mean|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id");
    println!(
        "  --factor-fill none|ffill (backtest default none; ffill supports low-frequency alpha)"
    );
    println!("  --exclude-limit true|false (backtest default true)");
    println!("  --exclude-st true|false (backtest default true)");
    println!("  --limit-side both|up|down (backtest default both)");
    println!("  --output-dir D:/path/to/output");
    println!(
        "  --factor-batch-size N (run default {}, backtest default {})",
        DEFAULT_FACTOR_BATCH_SIZE, DEFAULT_BACKTEST_FACTOR_BATCH_SIZE
    );
    println!(
        "  --date-batch-size N (run default {}, backtest default {})",
        DEFAULT_DATE_BATCH_SIZE, DEFAULT_BACKTEST_DATE_BATCH_SIZE
    );
    println!(
        "  --label-batch-size N (default {})",
        DEFAULT_LABEL_BATCH_SIZE
    );
    println!("  --label-batch-num N (alias of --label-batch-size)");
    println!("  --exposure-batch-size N (default 1 for one Barra family per batch)");
    println!("  --threads N");
    println!("  --profile");
    println!("  --refresh-minute-cache");
    println!("  --refresh-label-cache");
    println!("  --config D:/path/to/config.toml");
}

#[cfg(test)]
mod tests {
    use super::{
        flag_enabled, matches_tag_filter, parse_backtest_run_request, parse_barra_run_request,
        parse_csv_values, parse_flags, parse_label_run_request, parse_run_request, parse_yyyymmdd,
        FactorFill, LimitSide, DEFAULT_DATE_BATCH_SIZE, DEFAULT_LABEL_BATCH_SIZE,
    };

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

    #[test]
    fn list_tag_filter_requires_all_tags() {
        let row_tags = parse_csv_values("daily,FZZQ,neutralize");
        assert!(matches_tag_filter(
            &row_tags,
            &Some(parse_csv_values("FZZQ,daily"))
        ));
        assert!(!matches_tag_filter(
            &row_tags,
            &Some(parse_csv_values("FZZQ,missing"))
        ));
        assert!(matches_tag_filter(&row_tags, &None));
    }

    #[test]
    fn list_tag_filter_hides_deprecated_by_default() {
        let row_tags = parse_csv_values("daily,worldquant101alpha,deprecated");
        assert!(!matches_tag_filter(&row_tags, &None));
        assert!(!matches_tag_filter(
            &row_tags,
            &Some(parse_csv_values("worldquant101alpha"))
        ));
        assert!(matches_tag_filter(
            &row_tags,
            &Some(parse_csv_values("deprecated"))
        ));
    }

    #[test]
    fn label_batch_size_defaults_to_five_and_accepts_alias() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260401",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_label_run_request(&args, false).expect("request");
        assert_eq!(request.label_batch_size, DEFAULT_LABEL_BATCH_SIZE);

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260401",
            "--label-batch-num",
            "7",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_label_run_request(&args, false).expect("request");
        assert_eq!(request.label_batch_size, 7);
    }

    #[test]
    fn label_date_batch_size_defaults_to_one_and_accepts_flag() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_label_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, DEFAULT_DATE_BATCH_SIZE);

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--date-batch-size",
            "20",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_label_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, 20);
    }

    #[test]
    fn run_date_batch_size_defaults_to_one_and_accepts_flag() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, DEFAULT_DATE_BATCH_SIZE);

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--date-batch-size",
            "20",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, 20);
    }

    #[test]
    fn barra_date_batch_size_defaults_to_one_and_accepts_flag() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_barra_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, DEFAULT_DATE_BATCH_SIZE);

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--date-batch-size",
            "20",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_barra_run_request(&args, false).expect("request");
        assert_eq!(request.date_batch_size, 20);
    }

    #[test]
    fn backtest_universe_and_benchmark_have_defaults_and_accept_flags() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--factors",
            "utd",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_backtest_run_request(&args).expect("request");
        assert_eq!(request.universe, "mkt_all");
        assert_eq!(request.benchmark, "mkt_mean");
        assert!(request.exclude_limit);
        assert!(request.exclude_st);
        assert_eq!(request.limit_side, LimitSide::Both);
        assert_eq!(request.factor_fill, FactorFill::None);

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--factors",
            "utd",
            "--universe",
            "000300.SH",
            "--benchmark",
            "000905.SH",
            "--exclude-limit",
            "false",
            "--exclude-st",
            "false",
            "--limit-side",
            "up",
            "--factor-fill",
            "ffill",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_backtest_run_request(&args).expect("request");
        assert_eq!(request.universe, "000300.SH");
        assert_eq!(request.benchmark, "000905.SH");
        assert!(!request.exclude_limit);
        assert!(!request.exclude_st);
        assert_eq!(request.limit_side, LimitSide::Up);
        assert_eq!(request.factor_fill, FactorFill::ForwardFill);
    }

    #[test]
    fn backtest_external_factor_root_accepts_explicit_or_all_factors() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--factors",
            "ml_combo_alpha",
            "--factor-root",
            "data/ml_alpha",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_backtest_run_request(&args).expect("request");
        assert_eq!(
            request.factor_root.as_deref(),
            Some(std::path::Path::new("data/ml_alpha"))
        );
        assert_eq!(
            request.factor_ids.as_deref(),
            Some(&["ml_combo_alpha".to_string()][..])
        );

        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--all-factors",
            "--factor-root",
            "data/ml_alpha",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        let request = parse_backtest_run_request(&args).expect("request");
        assert!(request.all_factors);
        assert_eq!(
            request.factor_root.as_deref(),
            Some(std::path::Path::new("data/ml_alpha"))
        );
    }

    #[test]
    fn backtest_rejects_removed_write_detail_flag() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260424",
            "--factors",
            "utd",
            "--write-detail",
            "true",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        assert!(parse_backtest_run_request(&args).is_err());
    }

    #[test]
    fn label_batch_size_and_num_conflict_is_rejected() {
        let args = [
            "--asset",
            "stock",
            "--frequency",
            "daily",
            "--start-date",
            "20260401",
            "--end-date",
            "20260401",
            "--label-batch-size",
            "5",
            "--label-batch-num",
            "6",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
        assert!(parse_label_run_request(&args, false).is_err());
    }
}

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub project_config_path: PathBuf,
    pub data_root: PathBuf,
    pub factor_root: PathBuf,
    pub label_root: PathBuf,
    pub stock_sw_classification_path: PathBuf,
    pub stock_calendar_exchange: String,
    pub future_calendar_exchange: String,
}

impl EngineConfig {
    pub fn discover(config_path: Option<PathBuf>) -> Result<Self> {
        let config_path = match config_path {
            Some(path) => path,
            None => discover_project_config()?,
        };
        Self::from_project_config(config_path)
    }

    pub fn from_project_config(path: PathBuf) -> Result<Self> {
        let content = fs::read_to_string(&path)?;
        let data_root_value = parse_toml_string_value(&content, "base_data_dir")
            .ok_or_else(|| err("missing [paths].base_data_dir in project config"))?;
        let data_root = PathBuf::from(data_root_value);
        let factor_root = data_root.join("factors");
        let label_root = data_root.join("label");
        let stock_sw_classification_path =
            parse_toml_string_value(&content, "stock_sw_classification_path")
                .map(|value| normalize_data_path(&data_root, &value))
                .unwrap_or_else(|| {
                    data_root
                        .join("index_data")
                        .join("member_sw")
                        .join("sw_members.parquet")
                });
        Ok(Self {
            project_config_path: path,
            data_root,
            factor_root,
            label_root,
            stock_sw_classification_path,
            stock_calendar_exchange: "SSE".to_string(),
            future_calendar_exchange: "SHFE".to_string(),
        })
    }
}

fn discover_project_config() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let candidates = [
        cwd.join("config.toml"),
        cwd.join("..").join("config.toml"),
        cwd.join("..").join("..").join("config.toml"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(normalize_path(&candidate)?);
        }
    }
    Err(err(
        "could not discover config.toml; pass --config explicitly",
    ))
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        Ok(path.canonicalize()?)
    } else {
        Ok(path.to_path_buf())
    }
}

fn normalize_data_path(data_root: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        data_root.join(path)
    }
}

fn parse_toml_string_value(content: &str, key: &str) -> Option<String> {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('#') || !line.starts_with(key) {
            continue;
        }
        let (_, value) = line.split_once('=')?;
        let value = value.trim();
        if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
            return Some(value[1..value.len() - 1].to_string());
        }
    }
    None
}

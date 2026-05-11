use std::path::PathBuf;

use crate::core::{AssetClass, Frequency};
use crate::error::{err, Result};

pub const DEFAULT_BACKTEST_LABEL: &str = "future_vwap_return_1d";
pub const DEFAULT_UNIVERSE: &str = "mkt_all";
pub const DEFAULT_BENCHMARK: &str = "mkt_mean";
pub const DEFAULT_GROUPS: usize = 10;
pub const DEFAULT_FACTOR_BATCH_SIZE: usize = 10;
pub const DEFAULT_DATE_BATCH_SIZE: usize = 120;
pub const DEFAULT_EXCLUDE_LIMIT: bool = true;
pub const DEFAULT_EXCLUDE_ST: bool = true;

#[derive(Clone, Debug)]
pub struct BacktestRunRequest {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub factor_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub all_factors: bool,
    pub factor_root: Option<PathBuf>,
    pub label_id: String,
    pub groups: usize,
    pub rebalance: RebalanceRule,
    pub neutralize: NeutralizeSpec,
    pub universe: String,
    pub benchmark: String,
    pub exclude_limit: bool,
    pub exclude_st: bool,
    pub limit_side: LimitSide,
    pub factor_batch_size: usize,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
    pub factor_fill: FactorFill,
    pub output_dir: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactorFill {
    None,
    ForwardFill,
}

impl FactorFill {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" | "" => Ok(Self::None),
            "ffill" | "forward_fill" | "forward-fill" => Ok(Self::ForwardFill),
            _ => Err(err(format!(
                "--factor-fill must be none|ffill, got {value}"
            ))),
        }
    }

    pub fn is_forward_fill(&self) -> bool {
        matches!(self, Self::ForwardFill)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LimitSide {
    Both,
    Up,
    Down,
}

impl LimitSide {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "both" | "all" | "limit" => Ok(Self::Both),
            "up" | "limit_up" | "limit-up" => Ok(Self::Up),
            "down" | "limit_down" | "limit-down" => Ok(Self::Down),
            _ => Err(err(format!(
                "--limit-side must be both|up|down, got {value}"
            ))),
        }
    }

    pub fn allows(&self, is_limit_up: bool, is_limit_down: bool, is_limit: bool) -> bool {
        match self {
            Self::Both => !is_limit,
            Self::Up => !is_limit_up,
            Self::Down => !is_limit_down,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RebalanceRule {
    Daily,
    Every(usize),
    Weekly,
    Biweekly,
    Monthly,
    Quarterly,
}

impl RebalanceRule {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "daily" | "1" => Ok(Self::Daily),
            "weekly" | "week" => Ok(Self::Weekly),
            "biweekly" | "bi-weekly" | "2week" | "2weeks" => Ok(Self::Biweekly),
            "monthly" | "month" => Ok(Self::Monthly),
            "quarterly" | "quarter" => Ok(Self::Quarterly),
            other => {
                let days = other.parse::<usize>()?;
                if days == 0 {
                    return Err(err("--rebalance fixed-day value must be greater than 0"));
                }
                if days == 1 {
                    Ok(Self::Daily)
                } else {
                    Ok(Self::Every(days))
                }
            }
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Daily => "daily".to_string(),
            Self::Every(days) => days.to_string(),
            Self::Weekly => "weekly".to_string(),
            Self::Biweekly => "biweekly".to_string(),
            Self::Monthly => "monthly".to_string(),
            Self::Quarterly => "quarterly".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NeutralizeSpec {
    None,
    Sector,
    Barra {
        columns: Vec<String>,
        sector: bool,
        all: bool,
    },
}

impl NeutralizeSpec {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() || value.eq_ignore_ascii_case("none") {
            return Ok(Self::None);
        }
        if value.eq_ignore_ascii_case("sector") {
            return Ok(Self::Sector);
        }
        let Some(rest) = value.strip_prefix("barra:") else {
            return Err(err(format!(
                "--neutralize must be none|sector|barra:COL[,COL][+sector], got {value}"
            )));
        };
        let mut sector = false;
        let mut all = false;
        let mut columns = Vec::new();
        for part in rest
            .split('+')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            if part.eq_ignore_ascii_case("sector") {
                sector = true;
                continue;
            }
            for column in part
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
            {
                if matches!(
                    column.to_ascii_lowercase().as_str(),
                    "all" | "*" | "__all__"
                ) {
                    all = true;
                    continue;
                }
                columns.push(column.to_string());
            }
        }
        columns.sort();
        columns.dedup();
        if columns.is_empty() && !sector && !all {
            return Err(err(
                "--neutralize barra: requires at least one Barra column, all, or +sector",
            ));
        }
        Ok(Self::Barra {
            columns,
            sector,
            all,
        })
    }

    pub fn label(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Sector => "sector".to_string(),
            Self::Barra {
                columns,
                sector,
                all,
            } => {
                let mut label = if *all {
                    "barra_all".to_string()
                } else {
                    format!("barra_{}", columns.join("_"))
                };
                if *sector {
                    if columns.is_empty() && !*all {
                        label = "barra_sector".to_string();
                    } else {
                        label.push_str("_sector");
                    }
                }
                label
            }
        }
    }

    pub fn barra_columns(&self) -> Vec<String> {
        match self {
            Self::Barra { columns, .. } => columns.clone(),
            _ => Vec::new(),
        }
    }

    pub fn uses_sector(&self) -> bool {
        matches!(self, Self::Sector | Self::Barra { sector: true, .. })
    }

    pub fn uses_all_barra(&self) -> bool {
        matches!(self, Self::Barra { all: true, .. })
    }
}

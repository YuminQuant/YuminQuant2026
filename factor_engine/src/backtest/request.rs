use std::path::PathBuf;

use crate::core::{AssetClass, Frequency};
use crate::error::{err, Result};

pub const DEFAULT_BACKTEST_LABEL: &str = "future_vwap_return_1d";
pub const DEFAULT_GROUPS: usize = 10;

#[derive(Clone, Debug)]
pub struct BacktestRunRequest {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub factor_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub all_factors: bool,
    pub label_id: String,
    pub groups: usize,
    pub rebalance: RebalanceRule,
    pub neutralize: NeutralizeSpec,
    pub write_detail: bool,
    pub output_dir: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
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
    Industry,
    Barra {
        columns: Vec<String>,
        industry: bool,
    },
}

impl NeutralizeSpec {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() || value.eq_ignore_ascii_case("none") {
            return Ok(Self::None);
        }
        if value.eq_ignore_ascii_case("industry") || value.eq_ignore_ascii_case("sector") {
            return Ok(Self::Industry);
        }
        let Some(rest) = value.strip_prefix("barra:") else {
            return Err(err(format!(
                "--neutralize must be none|industry|barra:COL[,COL][+industry], got {value}"
            )));
        };
        let mut industry = false;
        let mut columns = Vec::new();
        for part in rest
            .split('+')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            if part.eq_ignore_ascii_case("industry") || part.eq_ignore_ascii_case("sector") {
                industry = true;
                continue;
            }
            for column in part
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
            {
                columns.push(column.to_string());
            }
        }
        columns.sort();
        columns.dedup();
        if columns.is_empty() && !industry {
            return Err(err(
                "--neutralize barra: requires at least one Barra column or +industry",
            ));
        }
        Ok(Self::Barra { columns, industry })
    }

    pub fn label(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Industry => "industry".to_string(),
            Self::Barra { columns, industry } => {
                let mut label = format!("barra_{}", columns.join("_"));
                if *industry {
                    if columns.is_empty() {
                        label = "barra_industry".to_string();
                    } else {
                        label.push_str("_industry");
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

    pub fn uses_industry(&self) -> bool {
        matches!(self, Self::Industry | Self::Barra { industry: true, .. })
    }
}

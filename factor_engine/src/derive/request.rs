use std::path::PathBuf;

use crate::core::AssetClass;

pub const DEFAULT_DERIVE_DATE_BATCH_SIZE: usize = 20;

#[derive(Clone, Debug)]
pub struct DeriveBarRequest {
    pub asset_class: AssetClass,
    pub source: BarSource,
    pub bar_size: usize,
    pub start_date: i32,
    pub end_date: i32,
    pub overwrite: bool,
    pub date_batch_size: usize,
    pub project_config_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarSource {
    Minute,
}

impl BarSource {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "minute" | "1m" | "minute_1m" => Some(Self::Minute),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minute => "minute",
        }
    }
}

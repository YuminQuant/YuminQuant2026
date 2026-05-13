use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{err, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StrategyAssetClass {
    Stock,
    Future,
    MultiAsset,
}

impl StrategyAssetClass {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "stock" => Some(Self::Stock),
            "future" => Some(Self::Future),
            "multi_asset" | "multi-asset" => Some(Self::MultiAsset),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stock => "stock",
            Self::Future => "future",
            Self::MultiAsset => "multi_asset",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarFrequency {
    Daily,
    Minute,
}

impl BarFrequency {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "daily" | "day" | "1d" => Some(Self::Daily),
            "minute" | "1m" | "minute_1m" => Some(Self::Minute),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Minute => "minute",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillPrice {
    NextOpen,
}

impl FillPrice {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "next_open" => Some(Self::NextOpen),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StrategyRunConfig {
    pub asset_class: StrategyAssetClass,
    pub strategy_id: String,
    pub strategy_class: String,
    pub start_date: i32,
    pub end_date: i32,
    pub initial_cash: f64,
    pub bar_frequency: BarFrequency,
    pub fill_price: FillPrice,
    pub commission_bps: f64,
    pub stamp_tax_bps: f64,
    pub slippage_bps: f64,
    pub lot_size: f64,
    pub data_root: PathBuf,
    pub model_root: PathBuf,
    pub factor_root: PathBuf,
    pub strategy_params: BTreeMap<String, String>,
}

impl StrategyRunConfig {
    pub fn load(path: &Path, data_root: PathBuf, factor_root: PathBuf) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let doc = SimpleToml::parse(&content);
        let asset_class = doc
            .string("", "asset_class")
            .and_then(|value| StrategyAssetClass::parse(&value))
            .ok_or_else(|| err("missing or invalid strategy asset_class"))?;
        let strategy_id = doc
            .string("", "strategy_id")
            .ok_or_else(|| err("missing strategy_id"))?;
        let strategy_class = doc
            .string("", "strategy_class")
            .ok_or_else(|| err("missing strategy_class"))?;
        let start_date = doc
            .i32("", "start_date")
            .ok_or_else(|| err("missing start_date"))?;
        let end_date = doc
            .i32("", "end_date")
            .ok_or_else(|| err("missing end_date"))?;
        let initial_cash = doc
            .f64("", "initial_cash")
            .ok_or_else(|| err("missing initial_cash"))?;
        let bar_frequency = doc
            .string("clock", "bar_frequency")
            .and_then(|value| BarFrequency::parse(&value))
            .ok_or_else(|| err("missing or invalid [clock].bar_frequency"))?;
        let fill_price = doc
            .string("execution", "fill_price")
            .and_then(|value| FillPrice::parse(&value))
            .unwrap_or(FillPrice::NextOpen);
        let model_root = doc
            .string("paths", "model_root")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root.join("models"));

        Ok(Self {
            asset_class,
            strategy_id,
            strategy_class,
            start_date,
            end_date,
            initial_cash,
            bar_frequency,
            fill_price,
            commission_bps: doc.f64("execution", "commission_bps").unwrap_or(0.0),
            stamp_tax_bps: doc.f64("execution", "stamp_tax_bps").unwrap_or(0.0),
            slippage_bps: doc.f64("execution", "slippage_bps").unwrap_or(0.0),
            lot_size: doc.f64("execution", "lot_size").unwrap_or(1.0).max(1.0),
            data_root,
            model_root,
            factor_root,
            strategy_params: doc.section_values("strategy"),
        })
    }

    pub fn output_dir(&self) -> PathBuf {
        self.data_root
            .join("strategy")
            .join(self.asset_class.as_str())
            .join(&self.strategy_id)
    }
}

#[derive(Default)]
struct SimpleToml {
    sections: BTreeMap<String, BTreeMap<String, String>>,
}

impl SimpleToml {
    fn parse(content: &str) -> Self {
        let mut doc = Self::default();
        let mut section = String::new();
        for raw in content.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].trim().to_string();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            doc.sections
                .entry(section.clone())
                .or_default()
                .insert(key.trim().to_string(), clean_value(value.trim()));
        }
        doc
    }

    fn string(&self, section: &str, key: &str) -> Option<String> {
        self.sections.get(section)?.get(key).cloned()
    }

    fn i32(&self, section: &str, key: &str) -> Option<i32> {
        self.string(section, key)?.parse().ok()
    }

    fn f64(&self, section: &str, key: &str) -> Option<f64> {
        self.string(section, key)?.parse().ok()
    }

    fn section_values(&self, section: &str) -> BTreeMap<String, String> {
        self.sections.get(section).cloned().unwrap_or_default()
    }
}

fn clean_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{BarFrequency, SimpleToml};

    #[test]
    fn simple_toml_reads_root_and_section_values() {
        let doc = SimpleToml::parse(
            r#"
            strategy_id = "s1"
            start_date = 20200101
            [clock]
            bar_frequency = "daily"
            "#,
        );
        assert_eq!(doc.string("", "strategy_id").as_deref(), Some("s1"));
        assert_eq!(doc.i32("", "start_date"), Some(20200101));
        assert_eq!(
            doc.string("clock", "bar_frequency")
                .and_then(|value| BarFrequency::parse(&value)),
            Some(BarFrequency::Daily)
        );
    }
}

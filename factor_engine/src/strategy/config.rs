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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FutureMarkPrice {
    Close,
    Settle,
}

impl FutureMarkPrice {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "close" => Some(Self::Close),
            "settle" => Some(Self::Settle),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FutureConfig {
    pub default_margin_ratio: f64,
    pub max_margin_ratio: f64,
    pub mark_price: FutureMarkPrice,
    pub margin_by_product: BTreeMap<String, f64>,
}

impl Default for FutureConfig {
    fn default() -> Self {
        Self {
            default_margin_ratio: 0.12,
            max_margin_ratio: 1.0,
            mark_price: FutureMarkPrice::Close,
            margin_by_product: BTreeMap::new(),
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
    pub buy_commission_bps: f64,
    pub sell_commission_bps: f64,
    pub short_commission_bps: f64,
    pub cover_commission_bps: f64,
    pub stamp_tax_bps: f64,
    pub slippage_bps: f64,
    pub lot_size: f64,
    pub data_root: PathBuf,
    pub model_root: PathBuf,
    pub factor_root: PathBuf,
    pub future: FutureConfig,
    pub detail: bool,
    pub market_products: Vec<String>,
    pub market_symbols: Vec<String>,
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
        let mut future = FutureConfig::default();
        future.default_margin_ratio = doc
            .f64("future", "default_margin_ratio")
            .unwrap_or(future.default_margin_ratio)
            .max(0.0);
        future.max_margin_ratio = doc
            .f64("future", "max_margin_ratio")
            .unwrap_or(future.max_margin_ratio)
            .clamp(0.0, 1.0);
        future.mark_price = doc
            .string("future", "mark_price")
            .and_then(|value| FutureMarkPrice::parse(&value))
            .unwrap_or(future.mark_price);
        future.margin_by_product = doc
            .section_values("future.margin_by_product")
            .into_iter()
            .filter_map(|(product, value)| {
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|ratio| ratio.is_finite() && *ratio >= 0.0)
                    .map(|ratio| (product.to_ascii_uppercase(), ratio))
            })
            .collect();

        let commission_bps = doc.f64("execution", "commission_bps").unwrap_or(0.0);
        let buy_commission_bps = doc
            .f64("execution", "buy_commission_bps")
            .or_else(|| doc.f64("execution", "long_commission_bps"))
            .unwrap_or(commission_bps);
        let sell_commission_bps = doc
            .f64("execution", "sell_commission_bps")
            .unwrap_or(commission_bps);
        let short_commission_bps = doc
            .f64("execution", "short_commission_bps")
            .unwrap_or(commission_bps);
        let cover_commission_bps = doc
            .f64("execution", "cover_commission_bps")
            .unwrap_or(commission_bps);
        let detail = parse_bool(doc.string("output", "detail").as_deref()).unwrap_or(false);
        let market_products = doc
            .string("market", "products")
            .map(|value| parse_csv_upper(&value))
            .unwrap_or_default();
        let market_symbols = doc
            .string("market", "symbols")
            .map(|value| parse_csv_upper(&value))
            .unwrap_or_default();

        Ok(Self {
            asset_class,
            strategy_id,
            strategy_class,
            start_date,
            end_date,
            initial_cash,
            bar_frequency,
            fill_price,
            commission_bps,
            buy_commission_bps,
            sell_commission_bps,
            short_commission_bps,
            cover_commission_bps,
            stamp_tax_bps: doc.f64("execution", "stamp_tax_bps").unwrap_or(0.0),
            slippage_bps: doc.f64("execution", "slippage_bps").unwrap_or(0.0),
            lot_size: doc.f64("execution", "lot_size").unwrap_or(1.0).max(1.0),
            data_root,
            model_root,
            factor_root,
            future,
            detail,
            market_products,
            market_symbols,
            strategy_params: doc.section_values("strategy"),
        })
    }

    pub fn output_dir(&self) -> PathBuf {
        self.data_root
            .join("strategy")
            .join(self.asset_class.as_str())
            .join(&self.strategy_id)
    }

    pub fn future_margin_ratio(&self, symbol: &str) -> f64 {
        let product = future_product_from_symbol(symbol);
        self.future
            .margin_by_product
            .get(&product)
            .copied()
            .unwrap_or(self.future.default_margin_ratio)
            .max(0.0)
    }

    pub fn commission_bps_for_side(&self, side: crate::strategy::order::OrderSide) -> f64 {
        match side {
            crate::strategy::order::OrderSide::Buy => self.buy_commission_bps,
            crate::strategy::order::OrderSide::Sell => self.sell_commission_bps,
            crate::strategy::order::OrderSide::Short => self.short_commission_bps,
            crate::strategy::order::OrderSide::Cover => self.cover_commission_bps,
        }
    }
}

pub fn future_product_from_symbol(symbol: &str) -> String {
    let head = symbol.split('.').next().unwrap_or(symbol);
    let product = head
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect::<String>();
    if product.is_empty() {
        head.to_ascii_uppercase()
    } else {
        product.to_ascii_uppercase()
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

fn parse_bool(value: Option<&str>) -> Option<bool> {
    match value?.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

fn parse_csv_upper(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_ascii_uppercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{parse_bool, parse_csv_upper, BarFrequency, SimpleToml};

    #[test]
    fn simple_toml_reads_root_and_section_values() {
        let doc = SimpleToml::parse(
            r#"
            strategy_id = "s1"
            start_date = 20200101
            [clock]
            bar_frequency = "daily"
            [market]
            products = "AG, IF"
            [output]
            detail = true
            [future.margin_by_product]
            IF = 0.12
            "#,
        );
        assert_eq!(doc.string("", "strategy_id").as_deref(), Some("s1"));
        assert_eq!(doc.i32("", "start_date"), Some(20200101));
        assert_eq!(
            doc.string("clock", "bar_frequency")
                .and_then(|value| BarFrequency::parse(&value)),
            Some(BarFrequency::Daily)
        );
        assert_eq!(
            doc.section_values("future.margin_by_product")
                .get("IF")
                .map(String::as_str),
            Some("0.12")
        );
        assert_eq!(
            parse_csv_upper(&doc.string("market", "products").unwrap()),
            vec!["AG", "IF"]
        );
        assert_eq!(
            parse_bool(doc.string("output", "detail").as_deref()),
            Some(true)
        );
    }
}

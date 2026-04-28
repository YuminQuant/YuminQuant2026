use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum AssetClass {
    Stock,
    Future,
}

impl AssetClass {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "stock" | "stocks" => Some(Self::Stock),
            "future" | "futures" => Some(Self::Future),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stock => "stock",
            Self::Future => "future",
        }
    }
}

impl Display for AssetClass {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Frequency {
    Daily,
    Minute1,
}

impl Frequency {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "daily" | "day" | "1d" => Some(Self::Daily),
            "minute_1m" | "1m" | "minute" => Some(Self::Minute1),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Minute1 => "minute_1m",
        }
    }
}

impl Display for Frequency {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DatasetId {
    StockDailyPv,
    StockDailyBasic,
    StockIncome,
    StockBalanceSheet,
    StockMinute1m,
    StockSwClassification,
    FutureDaily,
    FutureMinute1m,
}

impl DatasetId {
    pub fn asset_class(self) -> AssetClass {
        match self {
            Self::StockDailyPv
            | Self::StockDailyBasic
            | Self::StockIncome
            | Self::StockBalanceSheet
            | Self::StockMinute1m
            | Self::StockSwClassification => AssetClass::Stock,
            Self::FutureDaily | Self::FutureMinute1m => AssetClass::Future,
        }
    }

    pub fn frequency(self) -> Frequency {
        match self {
            Self::StockDailyPv
            | Self::StockDailyBasic
            | Self::StockIncome
            | Self::StockBalanceSheet
            | Self::StockSwClassification
            | Self::FutureDaily => Frequency::Daily,
            Self::StockMinute1m | Self::FutureMinute1m => Frequency::Minute1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::StockDailyPv => "stock.daily.pv",
            Self::StockDailyBasic => "stock.daily.basic",
            Self::StockIncome => "stock.income",
            Self::StockBalanceSheet => "stock.balancesheet",
            Self::StockMinute1m => "stock.minute.1m",
            Self::StockSwClassification => "stock.sw_classification",
            Self::FutureDaily => "future.daily",
            Self::FutureMinute1m => "future.minute.1m",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataRequest {
    pub dataset: DatasetId,
    pub columns: Vec<String>,
    pub financial_quarters: Option<usize>,
}

impl DataRequest {
    pub fn new(dataset: DatasetId, columns: &[&str]) -> Self {
        Self {
            dataset,
            columns: columns.iter().map(|value| value.to_string()).collect(),
            financial_quarters: None,
        }
    }

    pub fn financial_quarters(dataset: DatasetId, columns: &[&str], quarters: usize) -> Self {
        Self {
            dataset,
            columns: columns.iter().map(|value| value.to_string()).collect(),
            financial_quarters: Some(quarters),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct Lookback {
    pub trading_days: usize,
}

#[derive(Clone, Debug)]
pub struct FactorSpec {
    pub id: String,
    pub aliases: Vec<String>,
    pub name: String,
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub version: String,
    pub tags: Vec<String>,
    pub description: String,
    pub dependencies: Vec<DataRequest>,
    pub lookback: Lookback,
}

impl FactorSpec {
    pub fn output_column(&self) -> String {
        self.id.replace('.', "__").replace('-', "_")
    }

    pub fn registry_key(&self) -> String {
        factor_registry_key(self.asset_class.as_str(), self.frequency.as_str(), &self.id)
    }
}

pub fn factor_registry_key(asset_class: &str, frequency: &str, factor_id: &str) -> String {
    format!("{asset_class}|{frequency}|{factor_id}")
}

#[derive(Clone, Debug)]
pub struct FactorContext {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub load_start_date: i32,
    pub target_dates: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum FactorRowKey {
    Daily {
        trade_date: i32,
        ts_code: String,
    },
    Minute {
        trade_date: i32,
        trade_time: String,
        ts_code: String,
    },
}

impl FactorRowKey {
    pub fn trade_date(&self) -> i32 {
        match self {
            Self::Daily { trade_date, .. } | Self::Minute { trade_date, .. } => *trade_date,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FactorValue {
    pub key: FactorRowKey,
    pub value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FactorSeries {
    pub spec: FactorSpec,
    pub values: Vec<FactorValue>,
}

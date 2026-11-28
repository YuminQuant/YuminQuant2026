use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::artifact_io::StoragePathExt;
use crate::data::fund_portfolio::{
    FundPortfolioBatchIndex, FundPortfolioIndex, FUND_PORTFOLIO_COLUMNS,
};
use crate::data::parquet_io::read_parquet;
use crate::data::read_ahead::visit_filtered_inputs;
use crate::derive::storage::yearly_files;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;

use super::{columns, number, ratio, yoy};

struct Report {
    date: i32,
    year: Option<i32>,
    broker: usize,
    forecast: Option<f64>,
}

pub(super) struct AnalystHistory {
    by_stock: HashMap<String, Vec<Report>>,
}

impl AnalystHistory {
    pub(super) fn load(root: &Path, start: i32, end: i32) -> Result<Self> {
        let source = root.join("chn_stock_data/analyst_report");
        let files = yearly_files(&source)?;
        let paths: Vec<_> = files
            .range(start / 10_000..=end / 10_000)
            .map(|(_, path)| path.clone())
            .collect();
        let fields = columns(&["ts_code", "report_date", "org_name", "quarter", "np"]);
        let mut by_stock: HashMap<String, Vec<Report>> = HashMap::new();
        let mut brokers = HashMap::new();
        visit_filtered_inputs(
            &source,
            &paths,
            |path| read_parquet(path, Some(&fields)),
            |table| {
                let codes = table.required_utf8("ts_code")?;
                let dates = table.required_i32_date_cast("report_date")?;
                let names = table.required_utf8("org_name")?;
                let quarters = table.required_utf8("quarter")?;
                let forecasts = table.required_f64_cast("np")?;
                for i in 0..table.len {
                    let (Some(date), Some(code), Some(name)) =
                        (dates[i], codes[i].as_deref(), names[i].as_deref())
                    else {
                        continue;
                    };
                    let name = name.trim();
                    if !(start..=end).contains(&date) || name.is_empty() || code.is_empty() {
                        continue;
                    }
                    let year = quarters[i].as_deref().and_then(|quarter| {
                        let quarter = quarter.trim();
                        let fiscal_year = quarter.strip_suffix("Q4").unwrap_or(quarter);
                        (fiscal_year.len() == 4)
                            .then_some(fiscal_year)?
                            .parse::<i32>()
                            .ok()
                            .filter(|year| (1990..=2100).contains(year))
                    });
                    let next = brokers.len();
                    let broker = *brokers.entry(name.to_owned()).or_insert(next);
                    by_stock.entry(code.to_owned()).or_default().push(Report {
                        date,
                        year,
                        broker,
                        forecast: number(forecasts[i]),
                    });
                }
                Ok(())
            },
        )?;
        for reports in by_stock.values_mut() {
            reports.sort_by_key(|report| report.date);
        }
        Ok(Self { by_stock })
    }

    pub(super) fn metrics(
        &self,
        stock: &str,
        date: i32,
        window: Option<(i32, i32, i32)>,
    ) -> [Option<f64>; 5] {
        let Some((start, as_of, past_start)) = window else {
            return [None; 5];
        };
        let Some(reports) = self.by_stock.get(stock) else {
            return [Some(0.0), None, None, None, None];
        };
        let current_year = date / 10_000;
        let (current, next) = report_window(reports, start, date, current_year);
        let coverage = Some(current.len() as f64);
        let current_mean = mean(current.values().filter_map(|value| *value));
        let next_mean = mean(next.values().copied());
        let dispersion = next_mean.and_then(|average| {
            let scale = mean(next.values().map(|value| (value - average).powi(2)))?.sqrt();
            (next.len() > 1).then_some(())?;
            ratio(Some(scale), Some(average.abs()))
        });
        let (_, prior_next) = report_window(reports, past_start, as_of, current_year);
        let mut count = 0;
        let mut up = 0;
        let mut revised = 0.0;
        for (broker, value) in next {
            if let Some(previous) = prior_next.get(&broker) {
                if let Some(change) = yoy(Some(value), Some(*previous)) {
                    revised += change;
                    count += 1;
                    up += usize::from(change > 0.0);
                }
            }
        }
        let (revision, upgrade) = if count > 0 {
            (Some(revised / count as f64), Some(up as f64 / count as f64))
        } else {
            (None, None)
        };
        [
            coverage,
            dispersion,
            yoy(next_mean, current_mean),
            revision,
            upgrade,
        ]
    }
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0), |(sum, n), value| (sum + value, n + 1));
    (n > 0).then(|| sum / n as f64)
}

fn report_window(
    reports: &[Report],
    start: i32,
    end: i32,
    current: i32,
) -> (BTreeMap<usize, Option<f64>>, BTreeMap<usize, f64>) {
    let start = reports.partition_point(|report| report.date < start);
    let end = reports.partition_point(|report| report.date <= end);
    let mut current_values = BTreeMap::new();
    let mut next_values = BTreeMap::new();
    for report in &reports[start..end] {
        let value = current_values.entry(report.broker).or_insert(None);
        if let Some(forecast) = report.forecast {
            if report.year == Some(current) {
                *value = Some(forecast);
            } else if report.year == Some(current + 1) {
                next_values.insert(report.broker, forecast);
            }
        }
    }
    (current_values, next_values)
}

#[derive(Default)]
struct FundSnapshot {
    period: i32,
    announcement: i32,
    amounts: Vec<(u32, f64)>,
}

struct FundEvent {
    code: String,
    snapshot: FundSnapshot,
}

#[derive(Default)]
struct FundVersions {
    latest: FundSnapshot,
    previous: FundSnapshot,
}

#[derive(Default)]
pub(super) struct FundHistory {
    files: Vec<(i32, PathBuf)>,
    next_file: usize,
    end: i32,
    events: std::vec::IntoIter<FundEvent>,
    pending: Option<FundEvent>,
    funds: HashMap<String, FundVersions>,
    holdings: HashMap<u32, f64>,
    comparable_changes: HashMap<u32, f64>,
    comparable_fund_count: usize,
}

impl FundHistory {
    pub(super) fn load(root: &Path, end: i32) -> Result<Self> {
        let path = root.join("fund_data/portfolio");
        if !path.storage_exists() {
            eprintln!(
                "derive-fundamental-lfq: no fund portfolio under {}; fund columns will be null",
                path.display()
            );
            return Ok(Self::default());
        }
        let files = yearly_files(&path)?
            .into_iter()
            .filter(|(year, _)| *year <= end / 10_000)
            .collect();
        Ok(Self {
            files,
            end,
            ..Self::default()
        })
    }

    pub(super) fn advance(&mut self, date: i32) -> Result<()> {
        loop {
            if self.pending.is_none() {
                self.pending = self.events.next();
            }
            if let Some(event) = &self.pending {
                if event.snapshot.announcement > date {
                    break;
                }
                let event = self.pending.take().unwrap();
                self.update(event);
                continue;
            }
            let Some((year, path)) = self.files.get(self.next_file) else {
                break;
            };
            if *year > date / 10_000 {
                break;
            }
            let table = Arc::new(read_parquet(path, Some(&columns(FUND_PORTFOLIO_COLUMNS)))?);
            let index = FundPortfolioIndex::from_table(table)?;
            let batch = FundPortfolioBatchIndex::new(vec![Arc::new(index)], vec![], self.end);
            let reader = batch.reader();
            let mut events = Vec::new();
            // Compact each disclosure before releasing the original annual table.
            for group in reader.disclosures_between(None, self.end) {
                let mut amounts = BTreeMap::new();
                let mut valid = true;
                for &row in group.rows() {
                    let (Some(symbol), Some(shares)) = (
                        group.symbol(row).and_then(stock_key),
                        number(group.value(row, "amount")),
                    ) else {
                        continue;
                    };
                    if shares < 0.0 {
                        continue;
                    }
                    if amounts
                        .insert(symbol, shares)
                        .is_some_and(|previous| previous != shares)
                    {
                        valid = false;
                        break;
                    }
                }
                if valid && !amounts.is_empty() {
                    let event = FundEvent {
                        code: group.fund_code().to_owned(),
                        snapshot: FundSnapshot {
                            period: group.end_date(),
                            announcement: group.ann_date(),
                            amounts: amounts.into_iter().collect(),
                        },
                    };
                    if event.snapshot.announcement <= date {
                        self.update(event);
                    } else {
                        events.push(event);
                    }
                }
            }
            self.next_file += 1;
            self.events = events.into_iter();
        }
        Ok(())
    }

    fn update(&mut self, event: FundEvent) {
        let versions = self.funds.entry(event.code).or_default();
        let snapshot = event.snapshot;
        if (snapshot.period, snapshot.announcement)
            <= (versions.previous.period, versions.previous.announcement)
            || (snapshot.period == versions.latest.period
                && snapshot.announcement <= versions.latest.announcement)
        {
            return;
        }
        add_amounts(&mut self.holdings, &versions.latest.amounts, -1.0);
        if versions.comparable() {
            self.comparable_fund_count -= 1;
            add_amounts(&mut self.comparable_changes, &versions.latest.amounts, -1.0);
            add_amounts(
                &mut self.comparable_changes,
                &versions.previous.amounts,
                1.0,
            );
        }
        match snapshot.period.cmp(&versions.latest.period) {
            std::cmp::Ordering::Greater => {
                versions.previous = std::mem::replace(&mut versions.latest, snapshot)
            }
            std::cmp::Ordering::Equal => versions.latest = snapshot,
            std::cmp::Ordering::Less => versions.previous = snapshot,
        }
        add_amounts(&mut self.holdings, &versions.latest.amounts, 1.0);
        if versions.comparable() {
            self.comparable_fund_count += 1;
            add_amounts(&mut self.comparable_changes, &versions.latest.amounts, 1.0);
            add_amounts(
                &mut self.comparable_changes,
                &versions.previous.amounts,
                -1.0,
            );
        }
    }

    pub(super) fn metrics(&self, stock: &str, shares: Option<f64>) -> [Option<f64>; 2] {
        if self.funds.is_empty() {
            return [None, None];
        }
        let Some(stock) = stock_key(stock) else {
            return [None, None];
        };
        let shares = shares.filter(|value| *value > 0.0);
        let holding = Some(self.holdings.get(&stock).copied().unwrap_or(0.0).max(0.0));
        let change = (self.comparable_fund_count > 0)
            .then(|| self.comparable_changes.get(&stock).copied().unwrap_or(0.0));
        [ratio(holding, shares), ratio(change, shares)]
    }
}

impl FundVersions {
    fn comparable(&self) -> bool {
        self.previous.period > 0
            && previous_quarter_end_date(self.latest.period) == Some(self.previous.period)
    }
}

fn stock_key(code: &str) -> Option<u32> {
    let (digits, suffix) = code.trim().split_at_checked(6)?;
    if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let exchange = match suffix {
        ".SH" | ".sh" => 1,
        ".SZ" | ".sz" => 2,
        "" if digits.starts_with('6') => 1,
        "" if digits.starts_with(['0', '3']) => 2,
        _ => return None,
    };
    Some(exchange * 1_000_000 + digits.parse::<u32>().ok()?)
}

fn add_amounts(summary: &mut HashMap<u32, f64>, amounts: &[(u32, f64)], direction: f64) {
    for &(symbol, shares) in amounts {
        *summary.entry(symbol).or_default() += direction * shares;
    }
}

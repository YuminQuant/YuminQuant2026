use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::account::AccountState;
use crate::data::parquet_io::read_parquet;
use crate::error::Result;

#[derive(Clone, Debug)]
struct Distribution {
    symbol: String,
    ex_date: i32,
    pay_date: i32,
    listing_date: i32,
    stock_rate: f64,
    cash_rate: f64,
}

#[derive(Default)]
pub struct CorporateActions {
    records: BTreeMap<i32, Vec<Distribution>>,
    pending: Vec<(Distribution, f64, f64)>,
}

impl CorporateActions {
    pub fn load(root: &Path, start: i32, end: i32) -> Result<Self> {
        let mut book = Self::default();
        let folder = root.join("chn_stock_data/dividend");
        if !folder.exists() {
            eprintln!(
                "[strategy warning] missing stock dividend directory: {}",
                folder.display()
            );
            return Ok(book);
        }
        let mut paths = std::fs::read_dir(folder)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.extension().is_some_and(|s| s == "parquet"))
            .collect::<Vec<_>>();
        paths.sort();
        let mut seen = BTreeSet::new();
        for path in paths {
            let table = read_parquet(&path, None)?;
            let symbols = table.required_utf8("ts_code")?;
            let status = table.required_utf8("div_proc")?;
            let records = table.required_i32_date_cast("record_date")?;
            let ex_dates = table.required_i32_date_cast("ex_date")?;
            let pay_dates = table.required_i32_date_cast("pay_date")?;
            let listing = table.required_i32_date_cast("div_listdate")?;
            let stocks = table.required_f64_cast("stk_div")?;
            // Gross dividend: holding-period-dependent personal income tax is not inferred.
            let cash = table.required_f64_cast("cash_div_tax")?;
            for i in 0..table.len {
                if status[i].as_deref() != Some("实施") {
                    continue;
                }
                let (Some(symbol), Some(record), Some(ex_date)) =
                    (&symbols[i], records[i], ex_dates[i])
                else {
                    continue;
                };
                if record < start || record > end || ex_date <= record {
                    continue;
                }
                if !seen.insert((symbol.clone(), record, ex_date)) {
                    continue;
                }
                // Source rates are float32; normalize their decimal precision first.
                let rate = |value: Option<f64>| {
                    value
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .map(|v| (v * 1e6).round() / 1e6)
                        .unwrap_or(0.0)
                };
                let stock_rate = rate(stocks[i]);
                let cash_rate = rate(cash[i]);
                if stock_rate == 0.0 && cash_rate == 0.0 {
                    continue;
                }
                let pay_date = pay_dates[i].filter(|d| *d >= ex_date).unwrap_or_else(|| {
                    if cash_rate > 0.0 { eprintln!("[strategy warning] {symbol} ex_date={ex_date}: missing pay_date; using ex_date"); }
                    ex_date
                });
                let listing_date = listing[i].filter(|d| *d >= ex_date).unwrap_or_else(|| {
                    if stock_rate > 0.0 { eprintln!("[strategy warning] {symbol} ex_date={ex_date}: missing listing date; using ex_date"); }
                    ex_date
                });
                book.records.entry(record).or_default().push(Distribution {
                    symbol: symbol.clone(),
                    ex_date,
                    pay_date,
                    listing_date,
                    stock_rate,
                    cash_rate,
                });
            }
        }
        Ok(book)
    }

    pub fn begin_session(&mut self, account: &mut AccountState, date: i32) {
        account.settle_stock_entitlements(date);
        self.pending.retain(|(event, quantity, record_mark)| {
            if event.ex_date > date {
                return true;
            }
            account.apply_stock_distribution(
                &event.symbol,
                *quantity,
                event.stock_rate,
                event.cash_rate,
                event.pay_date,
                event.listing_date,
                date,
                *record_mark,
            );
            false
        });
    }

    pub fn end_session(&mut self, account: &AccountState, date: i32) {
        if let Some(events) = self.records.remove(&date) {
            for event in events {
                if let Some(position) = account.position(&event.symbol) {
                    if position.quantity > 0.0 {
                        self.pending
                            .push((event, position.quantity, position.mark_price()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_preserves_value_and_respects_payment_and_listing_dates() {
        let mut account = AccountState::new(20000.0);
        account.apply_buy("A", 1000.0, 10.0, 0.0);
        account.apply_stock_distribution("A", 1000.0, 0.5, 1.0, 20200108, 20200109, 20200107, 10.0);
        assert_eq!(account.position_quantity("A"), 1500.0);
        assert_eq!(account.sellable_quantity("A"), 1000.0);
        assert_eq!(account.dividend_receivable(), 1000.0);
        assert!((account.equity() - 20000.0).abs() < 1e-8);
        assert!((account.summary().equity - account.equity()).abs() < 1e-8);
        assert!(
            (account.account_pnl()
                - account.net_realized_pnl_cum()
                - account.total_unrealized_pnl())
            .abs()
                < 1e-8
        );
        account.apply_sell("A", 1000.0, 6.0, 0.0);
        account.settle_stock_entitlements(20200108);
        assert_eq!(account.cash(), 17000.0);
        assert_eq!(account.sellable_quantity("A"), 0.0);
        account.settle_stock_entitlements(20200109);
        assert_eq!(account.sellable_quantity("A"), 500.0);
        assert_eq!(account.dividend_receivable(), 0.0);
        assert!((account.equity() - 20000.0).abs() < 1e-8);
    }

    #[test]
    fn only_record_date_holders_receive_distribution() {
        let event = Distribution {
            symbol: "A".into(),
            ex_date: 20200107,
            pay_date: 20200108,
            listing_date: 20200107,
            stock_rate: 0.3,
            cash_rate: 0.045,
        };
        let mut book = CorporateActions::default();
        book.records.insert(20200106, vec![event]);
        let mut account = AccountState::new(100000.0);
        account.apply_buy("A", 1900.0, 14.28, 0.0);
        book.end_session(&account, 20200106);
        book.begin_session(&mut account, 20200107);
        assert_eq!(account.position_quantity("A"), 2470.0);
        assert!((account.dividend_receivable() - 85.5).abs() < 1e-8);
        account.apply_buy("A", 100.0, 11.0, 0.0);
        book.begin_session(&mut account, 20200107);
        assert_eq!(account.position_quantity("A"), 2570.0);
        assert!((account.dividend_receivable() - 85.5).abs() < 1e-8);
    }
}

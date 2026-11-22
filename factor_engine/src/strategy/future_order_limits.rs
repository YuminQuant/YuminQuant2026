//! Minimum order quantities are lots, not fut_basic.per_unit (physical units/lot).
//! Daily Tushare values take precedence over the dated fallback below.
use std::collections::BTreeMap;
use std::path::Path;

use super::config::future_product_from_symbol;
use crate::artifact_io::StoragePathExt;
use crate::data::parquet_io::read_parquet;
use crate::error::{err, Result};

pub(super) type DailyOrderLimits = BTreeMap<i32, BTreeMap<String, (f64, f64)>>;

pub(super) fn load(root: &Path, year: i32) -> Result<DailyOrderLimits> {
    let path = root
        .join("future_data/basic/order_limits")
        .join(format!("{year}.parquet"));
    if !path.storage_exists() {
        eprintln!(
            "[strategy warning] missing daily order limits {}; using dated fallback rules",
            path.display()
        );
        return Ok(BTreeMap::new());
    }
    let table = read_parquet(&path, None)?;
    if table.len == 0 {
        return Ok(BTreeMap::new());
    }
    let dates = table.required_i32_date_cast("trade_date")?;
    let symbols = table.required_utf8("ts_code")?;
    let opening = table.required_f64_cast("min_open_lots")?;
    let closing = table.required_f64_cast("min_close_lots")?;
    let mut limits = DailyOrderLimits::new();
    for i in 0..table.len {
        let (Some(date), Some(symbol), Some(open), Some(close)) =
            (dates[i], symbols[i].as_ref(), opening[i], closing[i])
        else {
            return Err(err("null daily order-limit row"));
        };
        if date / 10000 != year
            || !valid_lots(open)
            || !valid_lots(close)
            || limits
                .entry(date)
                .or_default()
                .insert(symbol.clone(), (open, close))
                .is_some()
        {
            return Err(err(format!(
                "invalid or duplicate daily order limits {date} {symbol}"
            )));
        }
    }
    Ok(limits)
}

fn valid_lots(value: f64) -> bool {
    value.is_finite() && value >= 1.0 && value.fract() == 0.0
}

/// Prefer the exact market identifier, then support legacy three-digit CZCE months.
pub(super) fn lookup<'a, T>(rows: &'a BTreeMap<String, T>, symbol: &str) -> Option<&'a T> {
    rows.get(symbol).or_else(|| {
        let alias = czce_alias(symbol)?;
        rows.get(&alias).or_else(|| {
            rows.iter()
                .find(|(key, _)| czce_alias(key).as_ref() == Some(&alias))
                .map(|(_, value)| value)
        })
    })
}

fn czce_alias(symbol: &str) -> Option<String> {
    let contract = symbol.strip_suffix(".ZCE")?;
    let start = contract.find(|ch: char| ch.is_ascii_digit())?;
    let month = &contract[start..];
    if !matches!(month.len(), 3 | 4) || !month.bytes().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(format!(
        "{}{}.ZCE",
        &contract[..start],
        &month[month.len() - 3..]
    ))
}

/// Announcement dates, contract scopes and source URLs are documented in
/// docs/FUTURE_ORDER_LIMITS.md. Dates are trading dates, including their night session.
pub(super) fn fallback(symbol: &str, date: i32) -> (f64, f64) {
    let product = future_product_from_symbol(symbol);
    let contract = symbol.split('.').next().unwrap_or(symbol);
    let month = contract
        .strip_prefix(&product)
        .filter(|month| month.len() == 4)
        .and_then(|month| month.parse::<i32>().ok())
        .map(|month| 200000 + month);
    let contract_range = |first, last| month.is_some_and(|month| (first..=last).contains(&month));
    let open = match product.as_str() {
        "PS" if date >= 20260403 && contract_range(202604, 202703) => 5.0,
        "PS" if date >= 20251229 && contract_range(202601, 202612) => 10.0,
        "PS" if date >= 20251222 && contract_range(202601, 202612) => 5.0,
        "PS" if (month == Some(202701) && date >= 20260114)
            || (month == Some(202702) && date >= 20260209)
            || (month == Some(202703) && date >= 20260310) =>
        {
            10.0
        }
        "LC" if date >= 20260922 && contract_range(202610, 202709) => 2.0,
        "LC" if date >= 20251226 && contract_range(202601, 202612) => 5.0,
        "LC" if (month == Some(202701) && date >= 20260114)
            || (month == Some(202702) && date >= 20260209)
            || (month == Some(202703) && date >= 20260310)
            || (month == Some(202704) && date >= 20260414) =>
        {
            5.0
        }
        "PT" | "PD"
            if (date >= 20251229 && matches!(month, Some(202606 | 202608 | 202610 | 202612)))
                || (date >= 20260209 && month == Some(202702)) =>
        {
            2.0
        }
        "CJ" if date >= 20230712 => 1.0,
        "CJ" if date >= 20221215 && !matches!(contract, "CJ301" | "CJ2301") => 2.0,
        "CJ" if date >= 20211216 => 4.0,
        "ZC" if date >= 20220309 => 4.0,
        "ZC" if date >= 20220222 => 2.0,
        "PM" if date >= 20220505 => 10.0,
        "WH" if date >= 20220510
            || (date >= 20220309
                && matches!(
                    contract,
                    "WH205"
                        | "WH207"
                        | "WH209"
                        | "WH211"
                        | "WH301"
                        | "WH2205"
                        | "WH2207"
                        | "WH2209"
                        | "WH2211"
                        | "WH2301"
                )) =>
        {
            10.0
        }
        "RI" | "JR" | "LR" if date >= 20220510 => 10.0,
        _ => 1.0,
    };
    (open, 1.0)
}

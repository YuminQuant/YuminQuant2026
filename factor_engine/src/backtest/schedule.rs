use crate::backtest::request::RebalanceRule;

pub fn rebalance_dates(dates: &[i32], rule: &RebalanceRule) -> Vec<i32> {
    match rule {
        RebalanceRule::Daily => dates.to_vec(),
        RebalanceRule::Every(step) => dates
            .iter()
            .enumerate()
            .filter_map(|(idx, date)| (idx % step == 0).then_some(*date))
            .collect(),
        RebalanceRule::Weekly => period_end_dates(dates, |date| week_index(date)),
        RebalanceRule::Biweekly => {
            let weekly = period_end_dates(dates, |date| week_index(date));
            weekly
                .into_iter()
                .enumerate()
                .filter_map(|(idx, date)| (idx % 2 == 0).then_some(date))
                .collect()
        }
        RebalanceRule::Monthly => period_end_dates(dates, |date| date / 100),
        RebalanceRule::Quarterly => period_end_dates(dates, |date| {
            let (year, month, _) = split_yyyymmdd(date);
            year * 10 + ((month - 1) / 3 + 1)
        }),
    }
}

pub fn date_after(dates: &[i32], date: i32, offset: usize) -> Option<i32> {
    let idx = dates.binary_search(&date).ok()?;
    dates.get(idx + offset).copied()
}

fn period_end_dates<F>(dates: &[i32], mut period_key: F) -> Vec<i32>
where
    F: FnMut(i32) -> i32,
{
    let mut output = Vec::new();
    for (idx, date) in dates.iter().copied().enumerate() {
        let current = period_key(date);
        let next = dates.get(idx + 1).copied().map(&mut period_key);
        if next != Some(current) {
            output.push(date);
        }
    }
    output
}

fn week_index(date: i32) -> i32 {
    let (year, month, day) = split_yyyymmdd(date);
    (days_from_civil(year, month, day) + 2).div_euclid(7)
}

fn split_yyyymmdd(date: i32) -> (i32, i32, i32) {
    let year = date / 10_000;
    let month = (date / 100) % 100;
    let day = date % 100;
    (year, month, day)
}

fn days_from_civil(year: i32, month: i32, day: i32) -> i32 {
    let y = year - i32::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe
}

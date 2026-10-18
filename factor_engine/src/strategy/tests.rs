use super::*;
use crate::data::parquet_io::{read_parquet, write_parquet};
use crate::data::{ColumnData, Table};
use crate::strategy::market::load_future_daily_frames;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "yq-strategy-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn config(&self) -> StrategyRunConfig {
        let path = self.0.join("strategy.toml");
        std::fs::write(
            &path,
            r#"asset_class = "future"
strategy_id = "regression"
strategy_class = "future::sma"
start_date = 20260105
end_date = 20260106
initial_cash = 1000000
[clock]
bar_frequency = "minute"
[market]
products = "AG"
[execution]
commission_bps = 2
slippage_bps = 1
lot_size = 1
[future]
default_margin_ratio = 0.1
max_margin_ratio = 0.5
[strategy]
product = "AG"
window = 2
capital_per_contract = 1000
add_quantity = 1
"#,
        )
        .unwrap();
        StrategyRunConfig::load(&path, self.0.clone(), self.0.join("factors")).unwrap()
    }
    fn engine(&self) -> StrategyEngine {
        let path = self.0.join("project.toml");
        std::fs::write(
            &path,
            format!("[paths]\nbase_data_dir = \"{}\"\n", self.0.display()),
        )
        .unwrap();
        StrategyEngine {
            config: EngineConfig::from_project_config(path).unwrap(),
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn market_table(
    dates: Vec<i32>,
    codes: Vec<Option<String>>,
    times: Vec<String>,
    prices: Vec<f64>,
) -> Table {
    let n = dates.len();
    let mut columns = BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(dates.into_iter().map(Some).collect()),
        ),
        ("ts_code".into(), ColumnData::Utf8(codes)),
        (
            "trade_time".into(),
            ColumnData::Utf8(times.into_iter().map(Some).collect()),
        ),
        ("vol".into(), ColumnData::F64(vec![Some(1000.0); n])),
        ("amount".into(), ColumnData::F64(vec![Some(100000.0); n])),
    ]);
    for name in ["open", "high", "low", "close", "settle"] {
        columns.insert(
            name.into(),
            ColumnData::F64(prices.iter().copied().map(Some).collect()),
        );
    }
    Table::new(columns).unwrap()
}

fn write_market(path: &Path, dates: &[i32], prices: &[f64]) {
    let n = dates.len();
    write_parquet(
        path,
        &market_table(
            dates.to_vec(),
            vec![Some("AG2606.SHF".into()); n],
            vec!["daily".into(); n],
            prices.to_vec(),
        ),
    )
    .unwrap();
}

fn bar(symbol: &str, close: f64) -> Bar {
    Bar {
        symbol: symbol.into(),
        trade_date: 20260105,
        trade_time: "09:00".into(),
        open: close,
        high: close,
        low: close,
        close,
        settle: None,
        multiplier: 10.0,
        volume: None,
        amount: None,
        is_limit_up: false,
        is_limit_down: false,
    }
}
fn market(bars: Vec<Bar>) -> MarketSnapshot {
    MarketSnapshot::new(
        BarEvent {
            trade_date: 20260105,
            trade_time: "09:00".into(),
            bar_frequency: BarFrequency::Minute,
            is_session_first: false,
            is_session_end: false,
        },
        bars,
    )
}

#[test]
fn summary_matches_existing_metrics_after_trades_and_marks() {
    let mut account = AccountState::new(100000.0);
    account.apply_buy("stock", 100.0, 10.0, 3.0);
    account.apply_derivative_trade("long", 2.0, 100.0, 1.0, 10.0, 0.1);
    account.apply_derivative_trade("short", -3.0, 200.0, 2.0, 20.0, 0.2);
    for price in [150.0, 80.0, 210.0] {
        account.mark_price("stock", price / 10.0);
        account.mark_price("long", price);
        account.mark_price("short", price);
        let summary = account.summary();
        assert_eq!(summary.equity, account.equity());
        assert_eq!(summary.unrealized_pnl, account.total_unrealized_pnl());
        assert_eq!(summary.gross_market_value, account.gross_market_value());
        assert_eq!(summary.net_market_value, account.net_market_value());
        assert_eq!(summary.margin_required, account.margin_required());
    }
    account.apply_derivative_trade("short", 3.0, 210.0, 1.0, 20.0, 0.2);
    assert_eq!(account.summary().equity, account.equity());
}

#[test]
fn held_positions_are_marked_each_bar_and_missing_quotes_retain_price() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let mut account = AccountState::new(100000.0);
    account.apply_derivative_trade("AG2606.SHF", 2.0, 100.0, 0.0, 10.0, 0.1);
    for (price, expected) in [(110.0, 200.0), (90.0, -200.0)] {
        mark_account_positions(
            &config,
            &mut account,
            &market(vec![bar("AG2606.SHF", price), bar("unheld", 999.0)]),
        );
        assert_eq!(account.total_unrealized_pnl(), expected);
        assert_eq!(account.positions().len(), 1);
    }
    mark_account_positions(&config, &mut account, &market(vec![]));
    mark_account_positions(
        &config,
        &mut account,
        &market(vec![bar("AG2606.SHF", f64::NAN)]),
    );
    assert_eq!(account.total_unrealized_pnl(), -200.0);
}

#[test]
fn annual_cache_reuses_file_prefers_daily_and_changes_year() {
    let fixture = Fixture::new();
    let annual = fixture.0.join("future_data/daily/2026.parquet");
    write_market(&annual, &[20260105, 20260106], &[100.0, 110.0]);
    let mut cache = FutureDailyCache::default();
    let meta = BTreeMap::new();
    assert_eq!(
        cache.load(&fixture.0, &[20260105], &meta).unwrap()[0].bars[0].close,
        100.0
    );
    // A second date must not reopen/decode the annual file.
    std::fs::write(&annual, b"invalid parquet after initial load").unwrap();
    assert_eq!(
        cache.load(&fixture.0, &[20260106], &meta).unwrap()[0].bars[0].close,
        110.0
    );
    write_market(
        &fixture.0.join("future_data/daily/2026/20260106.parquet"),
        &[20260106],
        &[120.0],
    );
    assert_eq!(
        cache.load(&fixture.0, &[20260106], &meta).unwrap()[0].bars[0].close,
        120.0
    );
    write_market(
        &fixture.0.join("future_data/daily/2027.parquet"),
        &[20270105],
        &[130.0],
    );
    assert_eq!(
        cache.load(&fixture.0, &[20270105], &meta).unwrap()[0].bars[0].close,
        130.0
    );
    let uncached = load_future_daily_frames(&fixture.0, &[20270105], &meta).unwrap();
    assert_eq!(uncached[0].bars[0].close, 130.0);
}

#[test]
fn parquet_symbol_filter_preserves_union_and_empty_schema() {
    let fixture = Fixture::new();
    let date = 20260105;
    let path = fixture.0.join("future_data/minute/2026/20260105.parquet");
    let table = market_table(
        vec![date; 4],
        vec![
            Some("IF2606.CFX".into()),
            Some("ag2606.SHF".into()),
            Some("CU2606.SHF".into()),
            None,
        ],
        vec![
            "08:00".into(),
            "09:00".into(),
            "10:00".into(),
            "11:00".into(),
        ],
        vec![100.0; 4],
    );
    write_parquet(&path, &table).unwrap();
    let frames = load_future_minute_frames(
        &fixture.0,
        &[date],
        &BTreeMap::new(),
        &["AG".into()],
        &["CU2606.SHF".into()],
    )
    .unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].event.trade_time, "09:00");
    assert!(frames[0].event.is_session_first);
    assert!(frames[1].event.is_session_end);
    let empty =
        load_future_minute_frames(&fixture.0, &[date], &BTreeMap::new(), &["AU".into()], &[])
            .unwrap();
    assert!(empty.is_empty());
    assert_eq!(
        load_future_minute_frames(&fixture.0, &[date], &BTreeMap::new(), &[], &[])
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn daily_output_matches_last_detailed_bar_and_full_day_counts() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let dates = [20260105, 20260106];
    write_parquet(
        &fixture.0.join("calendar/trade_cal_SSE.parquet"),
        &Table::new(BTreeMap::from([
            (
                "cal_date".into(),
                ColumnData::I32(dates.iter().copied().map(Some).collect()),
            ),
            ("is_open".into(), ColumnData::I64(vec![Some(1); 2])),
        ]))
        .unwrap(),
    )
    .unwrap();
    for date in dates {
        let prices = (0..300)
            .map(|i| 100.0 + ((i % 40) as f64 - 20.0).abs())
            .collect();
        let times = (0..300)
            .map(|i| format!("{:02}:{:02}", 9 + i / 60, i % 60))
            .collect();
        write_parquet(
            &fixture
                .0
                .join(format!("future_data/minute/2026/{date}.parquet")),
            &market_table(
                vec![date; 300],
                vec![Some("AG2606.SHF".into()); 300],
                times,
                prices,
            ),
        )
        .unwrap();
    }
    let engine = fixture.engine();
    let mut request = StrategyRunRequest {
        config_path: fixture.0.join("strategy.toml"),
        project_config_path: None,
        detail: Some(true),
    };
    let detail_report = engine.run(&request).unwrap();
    let detail = read_parquet(&detail_report.output_files[0], None).unwrap();
    assert_eq!(detail.len, 600); // Cross multiple streamed row groups.
    assert!(detail_report.trade_count > 0);
    let pnl = detail.required_f64_cast("unrealized_pnl").unwrap();
    assert!(pnl.windows(2).any(|pair| pair[0] != pair[1]));
    request.detail = Some(false);
    let daily_report = engine.run(&request).unwrap();
    assert_eq!(daily_report.trade_count, detail_report.trade_count);
    let daily = read_parquet(&config.output_dir().join("holdings.parquet"), None).unwrap();
    assert_eq!(daily.len, 2);
    for (name, column) in &daily.columns {
        if name.starts_with("trade_") && name.ends_with("_json") {
            assert_eq!(
                format!("{:?}", column),
                format!("{:?}", ColumnData::Utf8(vec![Some("[]".into()); 2]))
            );
        } else if ["buy_count", "sell_count", "short_count", "cover_count"].contains(&name.as_str())
        {
            let counts = detail.required_i64_cast(name).unwrap();
            let expected = counts
                .chunks(300)
                .map(|chunk| Some(chunk.iter().flatten().sum::<i64>()))
                .collect::<Vec<_>>();
            assert_eq!(daily.required_i64_cast(name).unwrap(), expected);
        } else {
            assert_eq!(
                format!("{:?}", column),
                format!("{:?}", detail.columns[name].take(&[299, 599])),
                "column {name}"
            );
        }
    }
}

#[test]
fn streaming_abort_preserves_previous_output_and_empty_output_has_schema() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let output_dir = config.output_dir();
    let mut writer = HoldingsWriter::new(&output_dir).unwrap();
    let account = AccountState::new(1000.0);
    let snapshot = holding_snapshot(&config, &account, &market(vec![]), &[]);
    writer.push(snapshot.clone()).unwrap();
    let path = writer.finish().unwrap();
    let before = std::fs::read(&path).unwrap();
    {
        let mut aborted = HoldingsWriter::new(&output_dir).unwrap();
        for _ in 0..300 {
            aborted.push(snapshot.clone()).unwrap();
        }
        // Simulate an error after a batch has already been written to the temp file.
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::read_dir(&output_dir).unwrap().count(), 1);
    HoldingsWriter::new(&output_dir).unwrap().finish().unwrap();
    let empty = read_parquet(&path, None).unwrap();
    assert_eq!(empty.len, 0);
    assert!(empty.columns.contains_key("account_pnl"));
}

fn stock_open_fixture(block_sale_on_rebalance: bool) -> (Fixture, StrategyRunRequest) {
    let fixture = Fixture::new();
    let dates = [
        20191231, 20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110,
    ];
    std::fs::write(
        fixture.0.join("strategy.toml"),
        format!(
            r#"asset_class = "stock"
strategy_id = "opening"
strategy_class = "stock::strategy_001"
start_date = 20200102
end_date = 20200110
initial_cash = 100000
[clock]
bar_frequency = "daily"
[execution]
commission_bps = 2
stamp_tax_bps = 5
slippage_bps = 5
lot_size = 100
[strategy]
signal_id = "score"
signal_root = "{}"
top_n = 1
rebalance_days = 5
cash_buffer = 0
"#,
            fixture.0.join("models").display()
        ),
    )
    .unwrap();
    write_parquet(
        &fixture.0.join("calendar/trade_cal_SSE.parquet"),
        &Table::new(BTreeMap::from([
            (
                "cal_date".into(),
                ColumnData::I32(dates.iter().copied().map(Some).collect()),
            ),
            (
                "is_open".into(),
                ColumnData::I64(vec![Some(1); dates.len()]),
            ),
        ]))
        .unwrap(),
    )
    .unwrap();
    for date in dates {
        let codes = vec![Some("A.SZ".to_string()), Some("B.SZ".to_string())];
        write_parquet(
            &fixture
                .0
                .join(format!("models/{}/{date}.parquet", date / 10000)),
            &Table::new(BTreeMap::from([
                ("trade_date".into(), ColumnData::I32(vec![Some(date); 2])),
                ("ts_code".into(), ColumnData::Utf8(codes.clone())),
                (
                    "score".into(),
                    ColumnData::F64(if date == 20191231 {
                        vec![Some(2.0), Some(1.0)]
                    } else {
                        vec![Some(1.0), Some(2.0)]
                    }),
                ),
            ]))
            .unwrap(),
        )
        .unwrap();
        let mut pv = market_table(
            vec![date; 2],
            codes.clone(),
            vec!["daily".into(); 2],
            vec![10.0, 20.0],
        );
        pv.columns.insert(
            "close".into(),
            ColumnData::F64(vec![Some(11.0), Some(21.0)]),
        );
        write_parquet(
            &fixture.0.join(format!(
                "chn_stock_data/daily/pv/{}/{date}.parquet",
                date / 10000
            )),
            &pv,
        )
        .unwrap();
        write_parquet(
            &fixture.0.join(format!(
                "chn_stock_data/daily/trade_filter/{}/{date}.parquet",
                date / 10000
            )),
            &Table::new(BTreeMap::from([
                ("trade_date".into(), ColumnData::I32(vec![Some(date); 2])),
                ("ts_code".into(), ColumnData::Utf8(codes)),
                ("is_limit_up".into(), ColumnData::Bool(vec![Some(false); 2])),
                (
                    "is_limit_down".into(),
                    ColumnData::Bool(vec![
                        Some(block_sale_on_rebalance && date == 20200109),
                        Some(false),
                    ]),
                ),
            ]))
            .unwrap(),
        )
        .unwrap();
    }
    let request = StrategyRunRequest {
        config_path: fixture.0.join("strategy.toml"),
        project_config_path: None,
        detail: None,
    };
    (fixture, request)
}

#[test]
fn stock_first_open_uses_previous_signal_and_rebalances_after_five_sessions() {
    let (fixture, request) = stock_open_fixture(false);
    let report = fixture.engine().run(&request).unwrap();
    let output = read_parquet(&report.output_files[0], None).unwrap();
    assert_eq!(
        output.required_i64_cast("buy_count").unwrap(),
        vec![
            Some(1),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(0)
        ]
    );
    assert_eq!(
        output.required_i64_cast("sell_count").unwrap(),
        vec![
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(0)
        ]
    );
    assert_eq!(
        output.required_utf8("symbols_json").unwrap()[0].as_deref(),
        Some("[\"A.SZ\"]")
    );
    assert_eq!(
        output.required_utf8("symbols_json").unwrap()[5].as_deref(),
        Some("[\"B.SZ\"]")
    );
    // 9,900 shares sized at open=10, rather than today's close=11.
    assert_eq!(
        output.required_utf8("quantities_json").unwrap()[0].as_deref(),
        Some("[9900]")
    );
    let cash = output.required_f64_cast("cash").unwrap();
    assert!(cash[0].unwrap() >= 0.0 && cash[0].unwrap() < 1000.7);
    assert!(cash[5].unwrap() >= 0.0 && cash[5].unwrap() < 2001.4);
}

#[test]
fn stock_frozen_sale_does_not_fund_new_buys_or_create_leverage() {
    let (fixture, request) = stock_open_fixture(true);
    let report = fixture.engine().run(&request).unwrap();
    let output = read_parquet(&report.output_files[0], None).unwrap();
    assert_eq!(output.required_i64_cast("sell_count").unwrap()[5], Some(0));
    assert_eq!(output.required_i64_cast("buy_count").unwrap()[5], Some(0));
    assert_eq!(
        output.required_utf8("symbols_json").unwrap()[5].as_deref(),
        Some("[\"A.SZ\"]")
    );
    assert!(output
        .required_f64_cast("cash")
        .unwrap()
        .iter()
        .all(|cash| cash.unwrap() >= 0.0));
}

#[test]
fn stock_top_n_blocked_slot_stays_cash_and_repeated_selection_is_round_tripped() {
    let (fixture, request) = stock_open_fixture(false);
    let path = fixture.0.join("strategy.toml");
    let cfg = std::fs::read_to_string(&path)
        .unwrap()
        .replace("top_n = 1", "top_n = 2");
    std::fs::write(&path, cfg).unwrap();
    let filter = fixture
        .0
        .join("chn_stock_data/daily/trade_filter/2020/20200102.parquet");
    let mut table = read_parquet(&filter, None).unwrap();
    table.columns.insert(
        "is_limit_up".into(),
        ColumnData::Bool(vec![Some(false), Some(true)]),
    );
    write_parquet(&filter, &table).unwrap();
    let report = fixture.engine().run(&request).unwrap();
    let output = read_parquet(&report.output_files[0], None).unwrap();
    // Half of the capital remains assigned to the unfilled B slot.
    assert_eq!(
        output.required_utf8("quantities_json").unwrap()[0].as_deref(),
        Some("[4900]")
    );
    assert!(output.required_f64_cast("cash").unwrap()[0].unwrap() > 50000.0);
    // A remains selected, but its prior batch must still close before a new purchase.
    assert_eq!(output.required_i64_cast("sell_count").unwrap()[5], Some(1));
    assert_eq!(output.required_i64_cast("buy_count").unwrap()[5], Some(2));
    assert_eq!(
        output.required_utf8("trade_symbols_json").unwrap()[5].as_deref(),
        Some("[\"A.SZ\",\"B.SZ\",\"A.SZ\"]")
    );
}

#[test]
fn stock_missing_signal_does_not_prevent_mature_batch_exit() {
    let (fixture, request) = stock_open_fixture(false);
    std::fs::remove_file(fixture.0.join("models/2020/20200108.parquet")).unwrap();
    let report = fixture.engine().run(&request).unwrap();
    let output = read_parquet(&report.output_files[0], None).unwrap();
    assert_eq!(output.required_i64_cast("sell_count").unwrap()[5], Some(1));
    assert_eq!(output.required_i64_cast("buy_count").unwrap()[5], Some(0));
    assert_eq!(
        output.required_i64_cast("position_count").unwrap()[5],
        Some(0)
    );
}

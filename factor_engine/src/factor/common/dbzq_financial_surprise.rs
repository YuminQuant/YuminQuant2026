use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorSeries, FactorSpec, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap, DailyPanel,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};

const PROFIT: &str = "n_income_attr_p";
const SHARES: &str = "total_share";
const CASH: &str = "c_fr_sale_sg";
const ASSETS: [&str; 4] = [
    "notes_receiv",
    "accounts_receiv",
    "prepayment",
    "inventories",
];
const EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
pub enum SurpriseKind {
    EpsJump,
    CashInflow,
}

impl SurpriseKind {
    fn id(self) -> &'static str {
        match self {
            Self::EpsJump => "eps_growth_jump",
            Self::CashInflow => "operating_cash_inflow_surprise",
        }
    }
    fn quarters(self) -> usize {
        match self {
            Self::EpsJump => 9,
            Self::CashInflow => 10,
        }
    }
    fn flow_dataset(self) -> DatasetId {
        match self {
            Self::EpsJump => DatasetId::StockIncome,
            Self::CashInflow => DatasetId::StockCashFlow,
        }
    }
    fn statement(self) -> FinancialStatementDataset {
        match self {
            Self::EpsJump => FinancialStatementDataset::Income,
            Self::CashInflow => FinancialStatementDataset::CashFlow,
        }
    }
}

pub fn spec(kind: SurpriseKind) -> FactorSpec {
    let (alias, name, description, flow_columns, balance_columns) = match kind {
        SurpriseKind::EpsJump => (
            "J", "EPS Growth Jump",
            "PIT J: single-quarter parent net profit / same-quarter end total shares proxies EPS. YoY uses absolute prior EPS. Fit the previous four YoY growths to times 1..4 without intercept, predict at 5, and divide the current surprise by sample std of the four training residuals. Requires nine complete quarters; no EPS mixing or share-action adjustment.",
            vec![PROFIT], vec![SHARES],
        ),
        SurpriseKind::CashInflow => (
            "O", "Operating Cash Inflow Surprise",
            "PIT O: single-quarter cash received from sales/services regressed on intercept, lagged quarterly cash inflow and current operating-asset change using eight prior complete quarters. Operating assets=notes receivable+accounts receivable+prepayments+inventories; change is QoQ. Current out-of-sample surprise / sample std of training residuals. Requires ten quarterly flow/balance observations; missing components are not zero-filled.",
            vec![CASH], ASSETS.to_vec(),
        ),
    };
    FactorSpec {
        id: kind.id().into(), aliases: vec![alias.into(), kind.id().to_ascii_uppercase()],
        name: name.into(), asset_class: AssetClass::Stock, frequency: Frequency::Daily,
        version: "0.1.0".into(),
        tags: ["DBZQ", "fundamental", "financial", "pit", "surprise", "sector_neutralize", "daily"].into_iter().map(str::to_string).collect(),
        description: format!("{description} Positive direction; SW level-1 sector-only neutralization; excludes BJ, no SIZE, winsorization or zscore."),
        dependencies: vec![
            DataRequest::financial_quarters(kind.flow_dataset(), &flow_columns, kind.quarters()),
            DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &balance_columns, kind.quarters()),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ], intraday_raw_dependencies: Vec::new(), lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute(kind: SurpriseKind, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let flow = data.financial_reader(
        kind.flow_dataset(),
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let raw = raw_panel(kind, &panel, &flow, &balance)?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    Ok(raw
        .cs_neutralize_regression_by_group(&[], None, |date, codes| sector.groups_for(date, codes))?
        .to_factor_series(spec(kind)))
}

fn raw_panel(
    kind: SurpriseKind,
    panel: &DailyPanel,
    flow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Result<PanelColumn> {
    // Batch-local instrument-keyed snapshots never reuse future PIT state backwards.
    let mut cache = InstrumentAlignedSnapshotCache::<f64>::default();
    let schedule = FinancialEventSchedule::from_pit_readers(&[flow.clone(), balance.clone()]);
    let n = panel.instruments().len();
    let mut snapshots = vec![None; n];
    let mut values = vec![None; panel.shape_len()];
    let mut last_date = None;
    for (day, date) in panel.dates().iter().copied().enumerate() {
        let start = day * n;
        let changed = day > 0
            && (0..n).any(|i| {
                panel.is_present_offset(start + i) != panel.is_present_offset(start + i - n)
            });
        if last_date.is_none() || changed || schedule.has_event_after_until(last_date, date) {
            snapshots = cached_financial_stock_snapshots_for_date(
                panel,
                date,
                &mut cache,
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| marker(kind, flow, balance, code, date),
                |date, code, _| snapshot(kind, flow, balance, code, date),
            );
        }
        for (i, value) in snapshots.iter().enumerate() {
            if panel.is_present_offset(start + i) {
                values[start + i] = *value;
            }
        }
        last_date = Some(date);
    }
    panel.column_from_values(values)
}

fn ends(kind: SurpriseKind, anchor: i32) -> Option<Vec<i32>> {
    let mut dates = vec![anchor];
    for _ in 1..kind.quarters() {
        dates.push(previous_quarter_end_date(*dates.last()?)?);
    }
    Some(dates)
}

fn marker(
    kind: SurpriseKind,
    flow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<FinancialEventMarker> {
    let mut marker = FinancialEventMarkerBuilder::new();
    for end in ends(kind, flow.latest_quarter_end_date(code, date)?)? {
        marker.include_reader_record_for_end_date(kind.statement(), flow, code, date, end);
        marker.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            code,
            date,
            end,
        );
    }
    marker.build()
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn snapshot(
    kind: SurpriseKind,
    flow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<f64> {
    let dates = ends(kind, flow.latest_quarter_end_date(code, date)?)?;
    let mut eps_values = [0.0; 9];
    let mut cash = [0.0; 10];
    let mut assets = [0.0; 10];
    for (idx, end) in dates.into_iter().enumerate() {
        let flow = flow.record_for_end_date(code, date, end)?;
        let balance = balance.record_for_end_date(code, date, end)?;
        match kind {
            SurpriseKind::EpsJump => {
                // Both profit and balance-sheet share capital use base units (yuan/shares).
                let shares = clean(balance.column(SHARES)).filter(|v| *v > EPS)?;
                eps_values[idx] = clean(Some(clean(flow.column(PROFIT))? / shares))?;
            }
            SurpriseKind::CashInflow => {
                cash[idx] = clean(flow.column(CASH))?;
                let mut total = 0.0;
                for column in ASSETS {
                    total += clean(balance.column(column))?;
                }
                assets[idx] = clean(Some(total))?;
            }
        }
    }
    match kind {
        SurpriseKind::EpsJump => eps_jump(&eps_values),
        SurpriseKind::CashInflow => cash_surprise(&cash, &assets),
    }
}

fn sample_std(values: &[f64]) -> Option<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    (sd.is_finite() && sd > EPS).then_some(sd)
}

// Arrays are newest first. Only the four PREVIOUS growths enter the fit.
fn eps_jump(eps: &[f64; 9]) -> Option<f64> {
    if !eps.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut growth = [0.0; 5];
    for idx in 0..5 {
        if eps[idx + 4].abs() <= EPS {
            return None;
        }
        growth[idx] = clean(Some((eps[idx] - eps[idx + 4]) / eps[idx + 4].abs()))?;
    }
    let scale = growth[1..].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if scale <= EPS {
        return None;
    }
    let history: [f64; 4] = std::array::from_fn(|i| growth[4 - i] / scale);
    let beta = history
        .iter()
        .enumerate()
        .map(|(i, g)| (i + 1) as f64 * g)
        .sum::<f64>()
        / 30.0;
    let residual: [f64; 4] = std::array::from_fn(|i| history[i] - beta * (i + 1) as f64);
    clean(Some(
        (growth[0] / scale - beta * 5.0) / sample_std(&residual)?,
    ))
}

fn cash_surprise(cash: &[f64; 10], assets: &[f64; 10]) -> Option<f64> {
    if !cash.iter().chain(assets).all(|v| v.is_finite()) {
        return None;
    }
    let y = std::array::from_fn(|i| cash[i + 1]);
    let x = std::array::from_fn(|i| [cash[i + 2], assets[i + 1] - assets[i + 2]]);
    ols_surprise(y, x, cash[0], [cash[1], assets[0] - assets[1]])
}

fn ols_surprise(y: [f64; 8], x: [[f64; 2]; 8], current_y: f64, current_x: [f64; 2]) -> Option<f64> {
    // Center to absorb the intercept; scale using training rows only. Two-column
    // QR avoids squaring the condition number in monetary-data normal equations.
    let ys = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let xs: [f64; 2] =
        std::array::from_fn(|j| x.iter().map(|v| v[j].abs()).fold(0.0_f64, f64::max));
    if ![ys, xs[0], xs[1]].iter().all(|v| v.is_finite() && *v > 0.0) {
        return None;
    }
    let y = y.map(|v| v / ys);
    let x = x.map(|v| [v[0] / xs[0], v[1] / xs[1]]);
    let ym = y.iter().sum::<f64>() / 8.0;
    let xm: [f64; 2] = std::array::from_fn(|j| x.iter().map(|v| v[j]).sum::<f64>() / 8.0);
    let yc = y.map(|v| v - ym);
    let xc = x.map(|v| [v[0] - xm[0], v[1] - xm[1]]);
    let r11 = xc.iter().map(|v| v[0] * v[0]).sum::<f64>().sqrt();
    if r11 <= EPS {
        return None;
    }
    let q1 = xc.map(|v| v[0] / r11);
    let r12 = (0..8).map(|i| q1[i] * xc[i][1]).sum::<f64>();
    let u2: [f64; 8] = std::array::from_fn(|i| xc[i][1] - r12 * q1[i]);
    let r22 = u2.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !r22.is_finite() || r22 <= 1e-10 {
        return None;
    }
    let c1 = (0..8).map(|i| q1[i] * yc[i]).sum::<f64>();
    let c2 = (0..8).map(|i| u2[i] / r22 * yc[i]).sum::<f64>();
    let b2 = c2 / r22;
    let b1 = (c1 - r12 * b2) / r11;
    let residual: [f64; 8] = std::array::from_fn(|i| yc[i] - b1 * xc[i][0] - b2 * xc[i][1]);
    let prediction = ym + b1 * (current_x[0] / xs[0] - xm[0]) + b2 * (current_x[1] / xs[1] - xm[1]);
    clean(Some((current_y / ys - prediction) / sample_std(&residual)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use crate::factor::common::FinancialPitIndex;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn j_formula_is_no_intercept_out_of_sample_and_sample_std() {
        let eps = [18.0, 6.0, 3.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        // Growth history oldest first: 1,3,2,5; beta=1.1, sample variance=0.9.
        let expected = 2.5 / 0.9_f64.sqrt();
        assert!((eps_jump(&eps).unwrap() - expected).abs() < 1e-12);
        assert!((eps_jump(&eps.map(|v| -v)).unwrap() + expected).abs() < 1e-12);
        assert!((eps_jump(&eps.map(|v| v * 100.0)).unwrap() - expected).abs() < 1e-12);
        let mut revised_current = eps;
        revised_current[0] = 20.0;
        assert!(
            (eps_jump(&revised_current).unwrap() - expected - 1.0 / 0.9_f64.sqrt()).abs() < 1e-12
        );
        assert!(eps_jump(&[1.0; 9]).is_none());
        for idx in 0..9 {
            let mut invalid = eps;
            invalid[idx] = f64::NAN;
            assert!(eps_jump(&invalid).is_none());
        }
        let mut zero = eps;
        zero[8] = 0.0;
        assert!(eps_jump(&zero).is_none());
    }

    #[test]
    fn o_qr_matches_known_coefficients_and_does_not_fit_current() {
        let x: [[f64; 2]; 8] = std::array::from_fn(|i| {
            [
                if i < 4 { -1.0 } else { 1.0 },
                if i % 4 < 2 { -1.0 } else { 1.0 },
            ]
        });
        let y = std::array::from_fn(|i| {
            10.0 + 2.0 * x[i][0] + 3.0 * x[i][1] + if i % 2 == 0 { -1.0 } else { 1.0 }
        });
        let sd = (8.0_f64 / 7.0).sqrt();
        assert!((ols_surprise(y, x, 26.0, [2.0, 3.0]).unwrap() - 3.0 / sd).abs() < 1e-12);
        assert!((ols_surprise(y, x, 27.0, [2.0, 3.0]).unwrap() - 4.0 / sd).abs() < 1e-12);
        assert!(
            (ols_surprise(
                y.map(|v| v * 1e8),
                x.map(|v| v.map(|v| v * 1e8)),
                26e8,
                [2e8, 3e8]
            )
            .unwrap()
                - 3.0 / sd)
                .abs()
                < 1e-12
        );
        assert!(ols_surprise(y, [[1.0; 2]; 8], 26.0, [2.0, 3.0]).is_none());
        assert!(ols_surprise(y, x.map(|v| [v[0], v[0] * 2.0]), 26.0, [2.0, 3.0]).is_none());
        let perfect = x.map(|v| 10.0 + 2.0 * v[0] + 3.0 * v[1]);
        assert!(ols_surprise(perfect, x, 26.0, [2.0, 3.0]).is_none());
    }

    #[test]
    fn o_quarter_indexing_matches_independent_least_squares() {
        let cash = [25.0, 13.0, 21.0, 14.0, 7.0, 19.0, 10.0, 17.0, 9.0, 12.0];
        let assets = [
            120.0, 100.0, 90.0, 95.0, 110.0, 94.0, 103.0, 98.0, 113.0, 105.0,
        ];
        // Reference: numpy.linalg.lstsq on eight historical rows with intercept.
        assert!((cash_surprise(&cash, &assets).unwrap() - 8.495227880752687).abs() < 1e-10);
        let mut invalid = cash;
        invalid[9] = f64::INFINITY;
        assert!(cash_surprise(&invalid, &assets).is_none());
    }

    type Row = (i32, i32, Vec<Option<f64>>);
    fn index(rows: &[Row], fields: &[&str], report_type: i64) -> FinancialPitIndex {
        let codes = ["000001.SZ", "600000.SH", "430001.BJ"];
        let keyed: Vec<_> = codes
            .iter()
            .flat_map(|code| rows.iter().map(move |row| (*code, row)))
            .collect();
        let mut columns = BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(keyed.iter().map(|r| Some(r.0.into())).collect()),
            ),
            (
                "end_date".into(),
                ColumnData::I32(keyed.iter().map(|r| Some(r.1 .0)).collect()),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(keyed.iter().map(|r| Some(r.1 .1)).collect()),
            ),
            (
                "f_ann_date".into(),
                ColumnData::I32(keyed.iter().map(|r| Some(r.1 .1)).collect()),
            ),
            (
                "report_type".into(),
                ColumnData::I64(vec![Some(report_type); keyed.len()]),
            ),
            (
                "update_flag".into(),
                ColumnData::I64(vec![Some(0); keyed.len()]),
            ),
        ]);
        for (i, field) in fields.iter().enumerate() {
            columns.insert(
                (*field).into(),
                ColumnData::F64(keyed.iter().map(|r| r.1 .2[i]).collect()),
            );
        }
        FinancialPitIndex::from_table(Arc::new(Table::new(columns).unwrap())).unwrap()
    }

    fn fixture(kind: SurpriseKind) -> (FinancialPitIndex, FinancialPitIndex) {
        let dates = ends(kind, 20250331).unwrap();
        let (flow_values, balance_values, fields, flow_field) = match kind {
            SurpriseKind::EpsJump => (
                vec![
                    1800.0, 600.0, 300.0, 400.0, 200.0, 100.0, 100.0, 100.0, 100.0,
                ],
                vec![vec![Some(100.0)]; 9],
                vec![SHARES],
                PROFIT,
            ),
            SurpriseKind::CashInflow => (
                vec![25.0, 13.0, 21.0, 14.0, 7.0, 19.0, 10.0, 17.0, 9.0, 12.0],
                [
                    120.0, 100.0, 90.0, 95.0, 110.0, 94.0, 103.0, 98.0, 113.0, 105.0,
                ]
                .map(|a| vec![Some(a), Some(0.0), Some(0.0), Some(0.0)])
                .to_vec(),
                ASSETS.to_vec(),
                CASH,
            ),
        };
        let flow_rows: Vec<_> = dates
            .iter()
            .enumerate()
            .map(|(i, d)| (*d, 20250501, vec![Some(flow_values[i])]))
            .collect();
        let mut balance_rows: Vec<_> = dates
            .iter()
            .enumerate()
            .map(|(i, d)| {
                (
                    *d,
                    if i == 0 { 20250505 } else { 20250501 },
                    balance_values[i].clone(),
                )
            })
            .collect();
        let mut revised = balance_rows.last().unwrap().clone();
        revised.1 = 20250602;
        revised.2[0] = revised.2[0].map(|v| v * 2.0);
        balance_rows.push(revised);
        (
            index(&flow_rows, &[flow_field], 2),
            index(&balance_rows, &fields, 1),
        )
    }

    #[test]
    fn both_surprises_pit_markers_batch_order_and_presence() {
        for kind in [SurpriseKind::EpsJump, SurpriseKind::CashInflow] {
            let (f, b) = fixture(kind);
            let flow = f.reader(ReportTypePreference::income_single_quarter());
            let balance = b.reader(ReportTypePreference::balance_sheet_consolidated());
            let code = "000001.SZ";
            assert!(snapshot(kind, &flow, &balance, code, 20250502).is_none());
            let before = snapshot(kind, &flow, &balance, code, 20250505).unwrap();
            let expected = match kind {
                SurpriseKind::EpsJump => 2.5 / 0.9_f64.sqrt(),
                SurpriseKind::CashInflow => 8.495227880752687,
            };
            assert!((before - expected).abs() < 1e-10);
            assert_ne!(
                marker(kind, &flow, &balance, code, 20250505),
                marker(kind, &flow, &balance, code, 20250602)
            );
            assert_ne!(
                Some(before),
                snapshot(kind, &flow, &balance, code, 20250602)
            );
            let dates = [20250502, 20250505, 20250506, 20250602];
            let codes = [code, "600000.SH", "430001.BJ"];
            let panel = |dates: &[i32], codes: &[&str]| {
                DailyPanel::from_index(
                    dates.to_vec(),
                    codes.iter().map(|c| c.to_string()).collect(),
                    &[20250602],
                    vec![true; dates.len() * codes.len()],
                )
                .unwrap()
            };
            let full = raw_panel(kind, &panel(&dates, &codes), &flow, &balance).unwrap();
            assert!(full.values()[..3].iter().all(Option::is_none));
            assert_eq!(full.values()[3], full.values()[6]);
            assert!((0..4).all(|d| full.values()[d * 3 + 2].is_none()));
            let reordered = raw_panel(
                kind,
                &panel(&dates[1..], &[codes[2], codes[0], codes[1]]),
                &flow,
                &balance,
            )
            .unwrap();
            for day in 0..3 {
                for (i, old) in [2, 0, 1].into_iter().enumerate() {
                    assert_eq!(
                        reordered.values()[day * 3 + i],
                        full.values()[(day + 1) * 3 + old]
                    );
                }
            }
            let mut present = vec![true; 12];
            present[3] = false;
            let changed = DailyPanel::from_index(
                dates.to_vec(),
                codes.map(str::to_string).to_vec(),
                &[20250602],
                present,
            )
            .unwrap();
            let changed = raw_panel(kind, &changed, &flow, &balance).unwrap();
            assert!(changed.values()[3].is_none());
            assert_eq!(changed.values()[6], full.values()[6]);
            let output = full.to_factor_series(spec(kind));
            assert_eq!(output.values.len(), 3);
            assert!(output.values.iter().all(|v| v.key.trade_date() == 20250602));
        }
    }

    #[test]
    fn both_surprises_metadata_and_sector_only_neutralization() {
        for kind in [SurpriseKind::EpsJump, SurpriseKind::CashInflow] {
            let spec = spec(kind);
            for tag in ["DBZQ", "fundamental"] {
                assert!(spec.tags.contains(&tag.into()));
            }
            assert!(!spec.dependencies.iter().any(|r| matches!(
                r.dataset,
                DatasetId::StockDailyPv
                    | DatasetId::StockDailyBasic
                    | DatasetId::StockBarraDaily
                    | DatasetId::StockConsensus
            )));
            assert_eq!(ends(kind, 20250331).unwrap().len(), kind.quarters());
        }
        let panel = DailyPanel::from_index(
            vec![20250602],
            ["000001.SZ", "000002.SZ", "600000.SH"]
                .map(str::to_string)
                .to_vec(),
            &[20250602],
            vec![true; 3],
        )
        .unwrap();
        let raw = panel
            .column_from_values(vec![Some(1.0), Some(3.0), Some(5.0)])
            .unwrap();
        let result = raw
            .cs_neutralize_regression_by_group(&[], None, |_, _| {
                vec![Some("A".into()), Some("A".into()), None]
            })
            .unwrap();
        assert_eq!(result.values(), &[Some(-1.0), Some(1.0), None]);
    }
}

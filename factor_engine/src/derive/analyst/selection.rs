use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Needs {
    pub revenue: bool,
    pub profit: bool,
    pub eps: bool,
    pub equity: bool,
    pub price: bool,
    pub rating: bool,
    pub target: bool,
    pub revisions: bool,
}

impl Needs {
    pub fn all() -> Self {
        Self {
            revenue: true,
            profit: true,
            eps: true,
            equity: true,
            price: true,
            rating: true,
            target: true,
            revisions: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Metric {
    Revenue,
    Profit,
    Eps,
    Na,
    Pb,
    Ps,
    Pe,
    Peg,
    Roe,
    RevenueYoy,
    ProfitYoy,
    Cagr,
}

impl Metric {
    fn parse(prefix: &str) -> Option<Self> {
        Some(match prefix {
            "con_or" => Self::Revenue,
            "con_np" => Self::Profit,
            "con_eps" => Self::Eps,
            "con_na" => Self::Na,
            "con_pb" => Self::Pb,
            "con_ps" => Self::Ps,
            "con_pe" => Self::Pe,
            "con_peg" => Self::Peg,
            "con_roe" => Self::Roe,
            "con_or_yoy" => Self::RevenueYoy,
            "con_np_yoy" => Self::ProfitYoy,
            "con_npcgrate_2y" => Self::Cagr,
            _ => return None,
        })
    }

    fn require(self, needs: &mut Needs) {
        use Metric::*;
        match self {
            Revenue | RevenueYoy => needs.revenue = true,
            Profit | ProfitYoy | Cagr => needs.profit = true,
            Eps => needs.eps = true,
            Na | Roe => {
                needs.profit = true;
                needs.equity = true;
            }
            Pe => {
                needs.price = true;
                needs.eps = true;
            }
            Peg => {
                Pe.require(needs);
                Cagr.require(needs);
            }
            Pb => {
                Pe.require(needs);
                Na.require(needs);
            }
            Ps => {
                Pe.require(needs);
                Profit.require(needs);
                Revenue.require(needs);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Output {
    Annual(Metric, usize),
    Roll(Metric),
    Revision(i32),
    Rating,
    Target,
}

impl Output {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "con_rating_strength" => return Some(Self::Rating),
            "con_target_price" => return Some(Self::Target),
            "con_npgrate_1w" => return Some(Self::Revision(7)),
            "con_npgrate_4w" => return Some(Self::Revision(28)),
            "con_npgrate_13w" => return Some(Self::Revision(91)),
            "con_npgrate_26w" => return Some(Self::Revision(182)),
            "con_npgrate_52w" => return Some(Self::Revision(364)),
            _ => {}
        }
        if let Some(prefix) = name.strip_suffix("_roll") {
            return Metric::parse(prefix).map(Self::Roll);
        }
        for i in 0..4 {
            if let Some(prefix) = name.strip_suffix(&format!("_fy{i}")) {
                return Metric::parse(prefix).map(|m| Self::Annual(m, i));
            }
        }
        None
    }
}

pub(super) struct OutputPlan {
    outputs: Vec<(String, Output)>,
    pub needs: Needs,
}

impl OutputPlan {
    pub fn new(columns: &[String]) -> Result<Self> {
        if columns.is_empty() {
            return Err(err("--columns must contain at least one consensus metric"));
        }
        let mut outputs = Vec::new();
        let mut names = BTreeSet::new();
        let mut needs = Needs::default();
        for name in columns {
            let output = Output::parse(name)
                .ok_or_else(|| err(format!("unknown consensus output column: {name}")))?;
            if !names.insert(name) {
                continue;
            }
            match output {
                Output::Annual(m, _) | Output::Roll(m) => m.require(&mut needs),
                Output::Revision(_) => {
                    needs.profit = true;
                    needs.revisions = true;
                }
                Output::Rating => needs.rating = true,
                Output::Target => needs.target = true,
            }
            outputs.push((name.clone(), output));
        }
        Ok(Self { outputs, needs })
    }

    pub fn build(
        &self,
        date: i32,
        market: &DailyMarketData,
        financial: &ConsensusFinancialData,
        state: &mut AnalystConsensusState,
    ) -> Result<Table> {
        let snapshot = market.snapshot(date);
        let mut values = vec![Vec::with_capacity(snapshot.rows.len()); self.outputs.len()];
        for row in &snapshot.rows {
            let years = fiscal_years(date, &row.ts_code, financial);
            let mut eval = Evaluator {
                code: &row.ts_code,
                date,
                price: effective_price(row.close, row.pre_close),
                financial,
                state,
                annual: HashMap::new(),
                roll: HashMap::new(),
            };
            for ((_, output), column) in self.outputs.iter().zip(&mut values) {
                let value = match *output {
                    Output::Annual(m, i) => eval.annual(m, years[i]),
                    Output::Roll(m) => eval.roll(m),
                    Output::Revision(days) => {
                        let current = eval.annual(Metric::Profit, years[0]);
                        yoy(
                            current,
                            eval.state
                                .previous_np_fy0(add_days(date, -days), &row.ts_code),
                        )
                    }
                    Output::Rating => compute_rating(&row.ts_code, date, eval.state).value,
                    Output::Target => compute_target_price(&row.ts_code, date, eval.state).value,
                };
                column.push(clean(value));
            }
        }
        let mut columns = BTreeMap::from([
            (
                "trade_date".into(),
                ColumnData::I32(vec![Some(date); snapshot.rows.len()]),
            ),
            (
                "ts_code".into(),
                ColumnData::Utf8(snapshot.rows.into_iter().map(|r| Some(r.ts_code)).collect()),
            ),
        ]);
        for ((name, _), values) in self.outputs.iter().zip(values) {
            columns.insert(name.clone(), ColumnData::F64(values));
        }
        Table::new(columns)
    }
}

// Memoized per-stock evaluation expands only the requested dependency branches.
struct Evaluator<'a> {
    code: &'a str,
    date: i32,
    price: Option<f64>,
    financial: &'a ConsensusFinancialData,
    state: &'a mut AnalystConsensusState,
    annual: HashMap<(Metric, i32), Option<f64>>,
    roll: HashMap<Metric, Option<f64>>,
}

fn peg(pe: Option<f64>, growth: Option<f64>) -> Option<f64> {
    match (pe, growth) {
        (Some(pe), Some(g)) if g > EPS && pe >= 0.0 => Some(pe / g),
        _ => None,
    }
}

impl Evaluator<'_> {
    fn annual(&mut self, metric: Metric, year: i32) -> Option<f64> {
        if let Some(value) = self.annual.get(&(metric, year)) {
            return *value;
        }
        use Metric::*;
        let value = match metric {
            Revenue | Profit | Eps | Na => {
                let base =
                    self.state
                        .annual_base_snapshot(self.code, self.date, year, self.financial);
                match metric {
                    Revenue => base.operating_revenue.value,
                    Profit => base.net_profit.value,
                    Eps => base.eps.value,
                    Na => base.net_assets,
                    _ => unreachable!(),
                }
            }
            Pe => safe_div(self.price, self.annual(Eps, year)),
            Pb | Ps => {
                let shares = safe_div(self.annual(Profit, year), self.annual(Eps, year));
                let denom = self.annual(if metric == Pb { Na } else { Revenue }, year);
                safe_div(self.price.zip(shares).map(|(p, s)| p * s), denom)
            }
            Roe => safe_div(self.annual(Profit, year), self.annual(Na, year)).map(|v| 100.0 * v),
            RevenueYoy | ProfitYoy => {
                let base = if metric == RevenueYoy {
                    Revenue
                } else {
                    Profit
                };
                yoy(self.annual(base, year), self.annual(base, year - 1))
            }
            Cagr => cagr_2y_abs_base_pct(self.annual(Profit, year), self.annual(Profit, year - 2)),
            Peg => peg(self.annual(Pe, year), self.annual(Cagr, year)),
        };
        self.annual.insert((metric, year), value);
        value
    }

    fn blend(&mut self, metric: Metric, year: i32) -> Option<f64> {
        weighted(
            self.annual(metric, year),
            self.annual(metric, year + 1),
            days_until_year_end(self.date) as f64 / 365.0,
        )
    }

    fn roll(&mut self, metric: Metric) -> Option<f64> {
        if let Some(value) = self.roll.get(&metric) {
            return *value;
        }
        use Metric::*;
        let year = self.date / 10_000;
        let value = match metric {
            Revenue | Profit | Eps | Na => self.blend(metric, year),
            Pe => safe_div(self.price, self.roll(Eps)),
            Pb | Ps => {
                let shares0 = safe_div(self.annual(Profit, year), self.annual(Eps, year));
                let shares1 = safe_div(self.annual(Profit, year + 1), self.annual(Eps, year + 1));
                let shares = weighted(
                    shares0,
                    shares1,
                    days_until_year_end(self.date) as f64 / 365.0,
                );
                let denom = self.roll(if metric == Pb { Na } else { Revenue });
                safe_div(self.price.zip(shares).map(|(p, s)| p * s), denom)
            }
            Roe => safe_div(self.roll(Profit).map(|v| v * 100.0), self.roll(Na)),
            RevenueYoy | ProfitYoy => {
                let base = if metric == RevenueYoy {
                    Revenue
                } else {
                    Profit
                };
                yoy(self.roll(base), self.blend(base, year - 1))
            }
            Cagr => cagr_2y_abs_base_pct(self.roll(Profit), self.blend(Profit, year - 2)),
            Peg => peg(self.roll(Pe), self.roll(Cagr)),
        };
        self.roll.insert(metric, value);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "synthetic compute benchmark; run explicitly with --ignored --nocapture"]
    fn analyst_consensus_selection_compute_benchmark() {
        let date = 20250703;
        let (mut market, financial, mut seed) = fixture(date);
        let template = seed
            .annual_base_snapshots
            .iter()
            .filter(|(key, _)| key.ts_code == "600000.SH")
            .map(|(key, value)| (key.year, *value))
            .collect::<Vec<_>>();
        seed.annual_base_snapshots.clear();
        let rows = &mut market.by_date.get_mut(&date).unwrap().rows;
        rows.clear();
        for i in 0..1000 {
            let code = format!("{i:06}.SZ");
            rows.push(MarketRow {
                ts_code: code.clone(),
                close: Some(40.0),
                pre_close: Some(39.0),
            });
            for (year, snapshot) in &template {
                seed.annual_base_snapshots.insert(
                    AnnualBaseSnapshotKey {
                        ts_code: code.clone(),
                        year: *year,
                    },
                    *snapshot,
                );
            }
        }
        let names = ["fy0", "fy1", "fy2", "fy3", "roll"]
            .into_iter()
            .flat_map(|suffix| {
                [
                    format!("con_npcgrate_2y_{suffix}"),
                    format!("con_peg_{suffix}"),
                ]
            })
            .collect::<Vec<_>>();
        let plan = OutputPlan::new(&names).unwrap();
        let mut full_state = seed.clone();
        let mut partial_state = seed;
        let now = std::time::Instant::now();
        for _ in 0..5 {
            std::hint::black_box(
                build_consensus_table_for_date(date, &market, &financial, &mut full_state).unwrap(),
            );
        }
        let full = now.elapsed();
        let now = std::time::Instant::now();
        for _ in 0..5 {
            std::hint::black_box(
                plan.build(date, &market, &financial, &mut partial_state)
                    .unwrap(),
            );
        }
        eprintln!("consensus synthetic 1000 stocks x 5 cached dates: full={full:?}, CAGR+PEG={:?}; excludes input/output IO", now.elapsed());
    }

    #[test]
    fn analyst_consensus_selection_dependencies_are_minimal() {
        let plan = OutputPlan::new(&["con_npcgrate_2y_roll".into()]).unwrap();
        assert!(plan.needs.profit);
        assert!(!plan.needs.price && !plan.needs.eps && !plan.needs.revenue);
        assert!(
            !plan.needs.equity && !plan.needs.revisions && !plan.needs.rating && !plan.needs.target
        );
        let peg = OutputPlan::new(&["con_peg_fy2".into()]).unwrap();
        assert!(peg.needs.price && peg.needs.eps && peg.needs.profit);
        assert!(!peg.needs.equity && !peg.needs.revenue && !peg.needs.revisions);
        assert!(OutputPlan::new(&[]).is_err());
        assert!(OutputPlan::new(&["con_unknown".into()]).is_err());
        assert!(OutputPlan::new(&["trade_date".into()]).is_err());
        assert_eq!(
            OutputPlan::new(&["con_pe_roll".into(), "con_pe_roll".into()])
                .unwrap()
                .outputs
                .len(),
            1
        );
    }

    fn fixture(
        date: i32,
    ) -> (
        DailyMarketData,
        ConsensusFinancialData,
        AnalystConsensusState,
    ) {
        let index = Arc::new(FinancialPitIndex::from_source_tables(Vec::new(), None).unwrap());
        let financial = ConsensusFinancialData {
            income_index: index.clone(),
            balance_index: index,
            needs: Needs::all(),
        };
        let mut market = DailyMarketData::default();
        let mut state = AnalystConsensusState::default();
        let codes = ["600000.SH", "000001.SZ"];
        market.by_date.insert(
            date,
            DailyMarketSnapshot {
                rows: codes
                    .iter()
                    .map(|code| MarketRow {
                        ts_code: code.to_string(),
                        close: Some(40.0),
                        pre_close: Some(39.0),
                    })
                    .collect(),
            },
        );
        for (i, code) in codes.iter().enumerate() {
            for year in 2021..=2028 {
                let profit = if year == 2023 && i == 1 {
                    None
                } else {
                    Some(((year - 2018) as f64).powi(2) * 10.0)
                };
                state.annual_base_snapshots.insert(
                    AnnualBaseSnapshotKey {
                        ts_code: code.to_string(),
                        year,
                    },
                    AnnualBaseSnapshot {
                        operating_revenue: BaseConsensus {
                            value: Some(1000.0 + (year - 2021) as f64 * 50.0),
                        },
                        net_profit: BaseConsensus { value: profit },
                        eps: BaseConsensus {
                            value: Some(1.0 + (year - 2021) as f64 * 0.1),
                        },
                        net_assets: Some(5000.0),
                        marker: AnnualBaseSnapshotMarker {
                            income: None,
                            balance: None,
                        },
                        valid_until: None,
                    },
                );
            }
            for days in [7, 28, 91, 182, 364] {
                state
                    .np_fy0_history
                    .entry(add_days(date, -days))
                    .or_default()
                    .insert(code.to_string(), 50.0 + days as f64);
            }
            state.ratings.entry(code.to_string()).or_default().insert(
                "org".into(),
                RatingObservation {
                    report_date: add_days(date, -1),
                    create_time: None,
                    strength: 0.75,
                },
            );
            state.targets.entry(code.to_string()).or_default().insert(
                "org".into(),
                TargetObservation {
                    report_date: add_days(date, -1),
                    create_time: None,
                    target_price: 60.0,
                },
            );
        }
        (market, financial, state)
    }

    #[test]
    fn analyst_consensus_selected_columns_match_full_generation() {
        for date in [20250430, 20250506, 20251231] {
            let (market, financial, state) = fixture(date);
            let full =
                build_consensus_table_for_date(date, &market, &financial, &mut state.clone())
                    .unwrap();
            let names = full
                .columns
                .keys()
                .filter(|s| s.as_str() != "trade_date" && s.as_str() != "ts_code")
                .cloned()
                .collect::<Vec<_>>();
            let all = OutputPlan::new(&names)
                .unwrap()
                .build(date, &market, &financial, &mut state.clone())
                .unwrap();
            for name in names {
                let single = OutputPlan::new(&[name.clone()])
                    .unwrap()
                    .build(date, &market, &financial, &mut state.clone())
                    .unwrap();
                assert_eq!(single.columns.len(), 3);
                let expected = full.required_f64_cast(&name).unwrap();
                for table in [&single, &all] {
                    let actual = table.required_f64_cast(&name).unwrap();
                    for (a, b) in actual.iter().zip(&expected) {
                        match (a, b) {
                            (Some(a), Some(b)) => assert!(
                                (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                                "{date} {name}: {a} != {b}"
                            ),
                            _ => assert_eq!(a, b, "{date} {name}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn analyst_consensus_disclosure_anchor_full_partial_and_roll_agree() {
        // A publishes early; B has a visible Q1 but a delayed annual report.
        // Null metrics still constitute disclosure. f_ann_date takes precedence.
        let income = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![
                    Some("600000.SH".into()),
                    Some("000001.SZ".into()),
                    Some("000001.SZ".into()),
                ]),
            ),
            (
                "end_date".into(),
                ColumnData::I32(vec![Some(20251231), Some(20251231), Some(20260331)]),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(vec![Some(20260301), Some(20260510), Some(20260420)]),
            ),
            (
                "f_ann_date".into(),
                ColumnData::I32(vec![Some(20260320), None, None]),
            ),
            ("report_type".into(), ColumnData::I64(vec![Some(1); 3])),
            ("update_flag".into(), ColumnData::I64(vec![Some(0); 3])),
            ("n_income_attr_p".into(), ColumnData::F64(vec![None; 3])),
        ]))
        .unwrap();
        let index = Arc::new(FinancialPitIndex::from_table(Arc::new(income)).unwrap());
        for (date, bases) in [
            (20260319, [2024, 2024]),
            (20260320, [2025, 2024]),
            (20260430, [2025, 2024]),
            (20260501, [2025, 2025]),
            (20270101, [2025, 2025]),
        ] {
            let (market, mut financial, mut state) = fixture(date);
            let baseline =
                build_consensus_table_for_date(date, &market, &financial, &mut state.clone())
                    .unwrap();
            financial.income_index = index.clone();
            // Synthetic per-calendar-year values stay fixed; only FY labels change.
            for (key, snapshot) in &mut state.annual_base_snapshots {
                snapshot.marker =
                    annual_base_snapshot_marker(&key.ts_code, date, key.year, &financial);
            }
            for (code, base) in ["600000.SH", "000001.SZ"].into_iter().zip(bases) {
                assert_eq!(
                    fiscal_years(date, code, &financial),
                    [base, base + 1, base + 2, base + 3]
                );
            }
            let full =
                build_consensus_table_for_date(date, &market, &financial, &mut state.clone())
                    .unwrap();
            let names = full
                .columns
                .keys()
                .filter(|s| s.as_str() != "trade_date" && s.as_str() != "ts_code")
                .cloned()
                .collect::<Vec<_>>();
            for name in &names {
                let single = OutputPlan::new(&[name.clone()])
                    .unwrap()
                    .build(date, &market, &financial, &mut state.clone())
                    .unwrap();
                assert_eq!(
                    single.required_f64_cast(name).unwrap(),
                    full.required_f64_cast(name).unwrap(),
                    "{date} {name}"
                );
                if name.ends_with("_roll")
                    || name == "con_rating_strength"
                    || name == "con_target_price"
                {
                    assert_eq!(
                        full.required_f64_cast(name).unwrap(),
                        baseline.required_f64_cast(name).unwrap(),
                        "unchanged {date} {name}"
                    );
                }
            }
            for i in 0..4 {
                let values = full.required_f64_cast(&format!("con_np_fy{i}")).unwrap();
                for (value, base) in values.into_iter().zip(bases) {
                    assert_eq!(value, Some(((base + i - 2018) as f64).powi(2) * 10.0));
                }
            }
            // History retains each observation date's own FY0 (existing revision convention).
            let previous_date = 20260319;
            let previous_base = fiscal_years(previous_date, "600000.SH", &financial)[0];
            state.remember_np_fy0(
                previous_date,
                HashMap::from([(
                    "600000.SH".into(),
                    ((previous_base - 2018) as f64).powi(2) * 10.0,
                )]),
            );
            if date == 20260320 {
                let revisions = compute_np_grates("600000.SH", 20260326, &state, Some(490.0));
                assert!(
                    (revisions.con_npgrate_1w.unwrap() - 100.0 * (490.0 - 360.0) / 360.0).abs()
                        < 1e-10
                );
            }
        }
    }

    #[test]
    fn analyst_consensus_cagr_evaluates_only_profit_dependencies() {
        let date = 20250703;
        let (_, financial, mut state) = fixture(date);
        let mut eval = Evaluator {
            code: "600000.SH",
            date,
            price: None,
            financial: &financial,
            state: &mut state,
            annual: HashMap::new(),
            roll: HashMap::new(),
        };
        assert!(eval.roll(Metric::Cagr).is_some());
        assert_eq!(eval.annual.len(), 4);
        assert!(eval
            .annual
            .keys()
            .all(|(metric, _)| *metric == Metric::Profit));
        assert_eq!(eval.roll.len(), 2);
    }
}

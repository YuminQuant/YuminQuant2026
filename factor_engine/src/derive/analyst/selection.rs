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

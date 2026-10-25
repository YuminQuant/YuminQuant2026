# Analyst Co-Coverage Formula Factors

This daily provider implements the formula factors in the Huatai reports dated
2022-07-07 and 2022-10-26. It does not implement their industry/sector controls,
event strategies or neural models. All outputs carry `HTZQ`, `analyst`,
`price_volume` and `cs_network`.

## Graph And Availability

- Use the stock-basic universe, excluding BJ. No PV panel anchor.
- A completed quarter's last trading session is its snapshot, usable at that
  session's close. A truncated date batch does not establish a quarter boundary.
- Read reports in `(snapshot - 6 calendar months, snapshot]`, projecting
  `ts_code`, `report_date`, `org_name` and `author_name`. The existing loader adds
  its required partition columns. No prediction values or financial PIT are needed.
- Analyst identity is institution plus individual author, using the existing
  relation derivation's coauthor delimiters. Duplicate analyst-stock memberships
  across dates, reports and forecast-year rows count once.
- Only stocks listed at the snapshot enter the graph. The network is frozen until
  the next snapshot; current missing peer signals are excluded and weights
  renormalized. Newly listed stocks wait for the next graph snapshot.
- `A_ij = ln(1 + number of common analysts)`, with zero diagonal.
- `M_ij = sum_k A_ik A_kj`, then set diagonal and all direct edges to zero.
- Keep one sparse quarterly graph in provider state, keyed by stock code. There
  is no graph or raw-factor disk cache and no dense indirect matrix.

For any signal `x` and validity indicator `v`, indirect weighted means use
`[A(Ax) - Bx] / [A(Av) - Bv]`, where `B` contains `A^2` only on the diagonal and
direct-edge support. The numerator and denominator use unnormalized first-pass
sums, not averages. A small relative denominator tolerance suppresses numerical
ghosts when indirect support is empty. Sparse support is built only for requests
that require indirect signals.

## Internal Inputs

Prices are `close * adj_factor`. Daily simple returns require both adjacent
calendar sessions' adjusted prices; do not bridge missing sessions. N-session
return compounds the finite daily returns in that window. Turnover is the mean
of valid `turnover_rate_f` values (percent); volatility is population standard
deviation of valid daily returns. All windows use `min_periods=1`, skip missing
observations and remain null when there are no valid samples. Return and
volatility also require a valid current price. These ordinary metrics are never
registered or written as factors.

## Outputs

Let `D(x)` and `I(x)` be direct and indirect weighted means, `R` be the own return,
`T` be mean turnover, `S` be return volatility, and `Z` be a cross-sectional
population z-score over valid analyst-covered stocks. Constant cross-sections
produce null z-scores.

| ID Prefix | Windows | Raw Formula |
| --- | --- | --- |
| `analyst_cf1_ret` | 20 | `D(R)`; July report's CF_RET |
| `analyst_cf_rev_raw` | 20 | `D(R) - R`; July report's raw CF_REV |
| `analyst_cf_turn` | 20 | `D(T) - T` |
| `analyst_cf_std` | 20 | `D(S) - S` |
| `analyst_cf2_ret` | 20 | `I(R)` |
| `analyst_cf_ret` | 20 | `Z(D(R)) + Z(I(R))` |
| `analyst_cf1_rev` | 20 | `Z(D(R)) - Z(R)` |
| `analyst_cf2_rev` | 20 | `Z(I(R)) - Z(R)` |
| `analyst_cf_rev` | 20 | `CF1_REV + CF2_REV` |

Append `_20` to each prefix, for nine independent outputs. No 10- or 40-session
variants are registered. Do not conflate
the two reports' different CF_REV definitions. Z-scores are computed before
final neutralization, not from already-neutralized sibling outputs.

The October report divides every indirect weight in row i by
`ln(number_of_direct_peers_i + 1)` and subsequently normalizes the row. This
scalar cancels. At the user's direction there is no adjusted-indirect output;
multilayer momentum and indirect/multilayer reversal use unadjusted indirect
momentum instead. The multilayer formulas are sums, not means.

## Processing And State

Each day, apply SW level-1 industry plus Barra SIZE neutralization to the raw
formula output. No additional winsorization, final z-score, missing-value fill
or sign reversal. Missing industry/SIZE stays null. No daily final-value hold:
only the graph is held; prices, formula outputs and regression update daily.

Only requested outputs, windows, metrics and two-hop support are computed.
Requests for turnover alone do not load prices/adjustment factors. Market
columns are panel-only and restricted to the requested window plus price anchor;
SIZE is requested only on target dates. The universe lookback also includes the
previous quarter boundary. Cross-batch graph reuse never caches final factor
values. An earlier or different snapshot rebuilds the graph safely.

Keep validation scripts, fixtures, isolated factor roots and profiling records
under ignored `tmp/`, never commit them. Validate sparse against explicit masked
matrix multiplication and compare the full output cross-section between a
120-day date-batch setting and one-day batches, including a quarter boundary.

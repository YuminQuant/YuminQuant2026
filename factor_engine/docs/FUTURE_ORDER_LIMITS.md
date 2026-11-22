# Futures order quantities

`fut_basic.trade_unit` and `per_unit` describe physical units per lot, not the
minimum number of lots in an opening order. The engine uses those fields for
the contract multiplier. Opening and closing order minimums are separate.

## Data precedence

With daily futures costs enabled, the engine first reads
`future_data/basic/order_limits/YYYY.parquet`. Each row contains `trade_date`,
`ts_code`, `min_open_lots` and `min_close_lots`; ingestion also retains
`source_ts_code` and `source`. Tushare's CZCE trade parameters use four-digit contract months; some historical
market identifiers use three digits.

The upstream snapshots remain at
`future_data/basic/trade_param/CZCE/YYYY/YYYYMMDD.parquet`.
API contract identifiers are retained unchanged. The engine prefers an exact
identifier match and also supports legacy three-digit CZCE month aliases.
The Tushare field is [`fut_trade_param.min_order_qty`](https://tushare.pro/document/2?doc_id=497).
The GFEX, CFFEX and DCE parameter endpoints inspected on 2026-11-22 do not
expose this field; `fut_basic` and `fut_settle` do not provide it either.
Missing historical rows use the dated fallback in `future_order_limits.rs`.
An API row always supersedes the fallback. Static TOML minimum-lot settings
can increase, but cannot reduce, the selected opening minimum.

## Reviewed fallback changes

Dates below are **trading dates**, including their preceding night session.
Other contracts start with the standard one-lot minimum. Closing minimums
remain one lot in these fallback rules. This is an order-size rule, not a
physical delivery unit or a position limit.

| Product | Effective trading date | Minimum opening lots | Source and scope |
|---|---:|---:|---|
| PS | 2025-12-22 | 5 | PS2601–PS2612; [exchange notice reproduced by Eastmoney](https://qhweb.eastmoney.com/news/list/1766065996052159.html) |
| PS | 2025-12-29 | 10 | PS2601–PS2612; [GFEX notice](https://www.gfex.com.cn/gfex/tzts/202512/f785caa8c818468cb3d26936f0cf3cd0.shtml) |
| PS | 2026-04-03 | 5 | PS2604–PS2703; [GFEX notice 2026/17, full text reproduced](https://www.xdsfutures.com.cn/company-news-details.aspx?id=1426) |
| LC | 2025-12-26 | 5 | LC2601–LC2612; [GFEX notice reproduced](https://www.gtjaqh.com/pc/a/cde75fc665cb35ac153b8e9b9000b5f2) |
| LC | 2026-09-22 | 2 | LC2610–LC2709; [GFEX notice](https://www.gfex.com.cn/gfex/tzts/202609/6a88d647eaae4c09afad1bc1043ab398.shtml) |
| PS/LC | 2026-01-14 | 10 / 5 | New PS2701/LC2701; [notice 2026/5, full text](https://chinaexchangecirculars.com/zh/notices/gfex-ac322dc14d263952e033/) |
| PS/LC | 2026-02-09 | 10 / 5 | New PS2702/LC2702; [notice 2026/10, full text](https://chinaexchangecirculars.com/zh/notices/gfex-da0a2033e17482902d5a/) |
| PS/LC | 2026-03-10 | 10 / 5 | New PS2703/LC2703; [notice 2026/13, full text](https://chinaexchangecirculars.com/zh/notices/gfex-77458e467899c7ad4e25/) |
| LC | 2026-04-14 | 5 | New LC2704; [listing notice](https://chinaexchangecirculars.com/zh/parameters/gfex/lc/) |
| PT/PD | 2025-12-29 | 2 | 2606/2608/2610/2612; [notice reproduced](https://www.jyqh.com.cn/notice/1442) |
| PT/PD | 2026-02-09 | 2 | New 2702; [notice 2026/10](https://chinaexchangecirculars.com/zh/notices/gfex-da0a2033e17482902d5a/) |
| CJ | 2021-12-16 | 4 | [CZCE notice 2021/108](http://www.czce.com.cn/cn/gyjys/jysdt/ggytz/webinfo/2021/12/1640080563921774.htm) |
| CJ | 2022-12-15 | 2 | CJ2301 remains 4; [CZCE notice 2022/98](https://www.czce.com.cn/cn/rootfiles/2023/01/13/1655833212264330-1655833212279721.pdf) |
| CJ | 2023-07-12 | 1 | [CZCE notice 2023/53](http://www.czce.com.cn/cn/gyjys/jysdt/ggytz/webinfo/2023/07/1689459021126759.htm) |
| ZC | 2022-02-22 | 2 | Applies from the night session of Feb 21; [CZCE notice 2022/8](https://www.czce.com.cn/cn/rootfiles/2023/02/14/1655835976554752-1655835976576650.pdf) |
| ZC | 2022-03-09 | 4 | Applies from the night session of Mar 8; [exchange notice reproduced](https://stcn.com/article/detail/537090.html) |
| WH | 2022-03-09 | 10 | WH2205/2207/2209/2211/2301; [same notice](https://stcn.com/article/detail/537090.html) |
| PM | 2022-05-05 | 10 | [CZCE notice 2022/31](http://www.czce.com.cn/cn/gyjys/jysdt/ggytz/webinfo/2022/04/1644404850628802.htm) |
| WH/RI/JR/LR | 2022-05-10 | 10 | [exchange rule summary](https://www.founderfu.com/fzzqqh_2019/service_jygz_ty_257.html) |

GFEX fallbacks are restricted to the contract ranges listed above. Later
contract listings need their own reviewed rule or an API observation; a
future announcement must not be applied to earlier trading dates.

## Matching

CTA opens one selected minimum opening unit. Liquidation uses the closing
minimum independently, so a partially closed five-lot position can still
close its remaining one or two lots. When price-limit checking is enabled,
available daily upper/lower prices restrict matching. A missing file, date
or contract price produces a warning but does not restrict trading for the
missing limit, as required by the research execution convention.

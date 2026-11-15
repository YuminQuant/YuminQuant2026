# liquidity_001 期货主力流动性池 / Futures Main-Contract Liquidity Pool

## 筛选口径

1. 在上一年最后一个交易日，以该日为窗口终点，取最近 240 个期货交易日。
2. 对每个品种，从现有主力映射确定每天的实际主力合约，取该合约的日成交量 `vol`。要求 240 日主力成交量记录完整，以此判定上市交易历史满 240 日。有效的零成交量计入均值，缺失值不填零。
3. 保留日均成交量至少 10,000 手的品种，按日均成交量降序取前 50 名。同量时按品种代码升序排列。符合条件的品种不足 50 个时，保留全部符合条件的品种。
4. 入池品种在下一年固定。每日输出当日原主力映射中的实际合约，保留原合约编码。某品种当日无主力映射时缺该行，整日无映射时跳过并报告该日。

每个 `YYYY.parquet` 包含该年生成的 `trade_date`（`YYYYMMDD`）、`ts_code`（品种代码）和 `mapping_ts_code`（实际主力合约代码）。补跑只覆盖本次日期范围内有原主力映射的交易日。若上一年用于排名的数据有修订，需重算受影响的整年。

## Selection rules

1. Use the 240 futures trading sessions ending on the final trading session of the previous year, inclusive.
2. For each product and session, resolve the actual main contract from the existing main-contract mapping and use that contract's daily volume (`vol`). A complete 240-session main-contract volume history establishes the minimum listing history. Valid zero volumes enter the average. Missing values stay missing.
3. Keep products averaging at least 10,000 lots per session and select up to 50 in descending average-volume order. Break ties by ascending product code. Keep all eligible products when fewer than 50 qualify.
4. Membership is fixed for the following year. For each session, emit the actual main contract using the existing mapping and symbol format. A missing product mapping produces no row for that product. A session with no mapping is skipped and reported.

Each `YYYY.parquet` contains `trade_date` (`YYYYMMDD`), `ts_code` (product code), and `mapping_ts_code` (actual main-contract code). Reruns replace only requested sessions with source mappings. If the previous year's ranking inputs are revised, regenerate the full affected year.

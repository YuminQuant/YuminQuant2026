# 四个 Bar GRU 与华泰人工智能 68 研报核对

依据：2023-05-11《神经网络多频率因子挖掘模型》，第 5 页图表 7、第 6 页图表 8–9、
第 7 页图表 11、第 8 页图表 12–13、第 10 页集成与回测说明。

## 输入与网络

| 因子 | 输入 | 网络与目标 | 核对结果 |
| --- | --- | --- | --- |
| bar_gru_15m | 20 日 × 16 根 15 分钟 bar × 6 特征 | 单层 GRU(30) → BN → FC(1)，负 IC | 原实现符合 |
| bar_gru_attention_15m | 同上，320×6 | GRU(30) 全序列时间注意力，加权向量与末端隐状态拼接 60 维 → BN → FC(1)，负 IC | 按第 6 页公式及图表新增 |
| multi_bar_gru_daily_15m | 日频 40×6，分钟频 320×6 | 两个 GRU(30) 各自 BN，拼接 60 维 → FC(1)，负 IC | 原实现符合 |
| residual_multi_bar_gru | 同上 | 先训练日频 GRU→BN→FC；冻结日频，再训练分钟分支，以两分支输出之和计算负 IC | 原训练目标符合；补强冻结状态 |

六特征为 open、high、low、close、vwap、volume。分钟序列逐特征除以
序列均值，日频序列逐特征除以最后一个值，之后按当前股票截面逐位置 z-score；
标签同样做截面 z-score。特征窗口截止样本日，不读未来输入。
研报未明确复权价格等细节，沿用现有行情 provider，不声称逐数值复现。

Attention 版本逐时间步计算标量 `u_i=tanh(Linear(z_i))`，然后沿 320 个时间步
做 softmax，得到 `summary=sum_i(weight_i*z_i)`。每只股票独立计算注意力，
权重不跨股票或交易日；拼接 `[last_hidden, summary]` 后进行 BN(60) 和 FC(60,1)。
这是研报显示公式的一种直接实现，不引入额外的注意力隐藏层或第二个打分投影。
模型继承 `BarGRUAlphaModel`，仅替换网络和补充结构诊断，沿用同一输入 provider、
负 IC 训练、完整日期验证、预测及 checkpoint 保存/加载流程，不修改其他三个模型。

Residual 第二阶段不是另行拟合残差 MSE，而是最小化
`-PearsonIC(y_daily + y_minute, label)`。日频参数、BN 运行均值/方差都须冻结；
现在即使调用父模型 `train()`，日频分支仍保持 eval 状态。

## 修正与用户指定差异

- 原三个配置使用 5 日标签，改为研报的 10 日收益周期。研报没有明确标签成交价格，
  本项目选用 `future_vwap_return_10d`，即 VWAP(t+11)/VWAP(t+1)-1；
  `lookahead_days=11`，由公共层落实到期约束。
- 原固定训练和 10 日采样改为年度滚动、36 个月回看、33/3 月划分、5 交易日采样。
  这是用户统一规范，覆盖研报半年重训、训练验证 4:1 的设置。
- 按用户要求保留 `batch_size=3000` 和单种子 `seed=42`；
  与研报 batch 5000、三种子等权集成有意不同，不增加集成。
- 保留 Adam、学习率 0.001、最大 100 轮、早停 10 轮；Residual 两阶段分别执行。
- 验证前向统一受 batch_size 限制，将预测搬回 CPU 后拼接，按完整日期计算 IC，
  再对日期取均值；不是分块 IC 的平均。训练仍按单日股票截面分批计算负 IC。
- 研报回测阶段的去极值、行业/市值中性化等不直接塞进网络；
  当前模型返回原始预测分数，评估处理需在回测口径中独立确认。

Attention 新配置沿用上述项目口径：`batch_size=3000`、单种子 42、
Adam lr=0.001、最多 100 轮、验证早停 10 轮；不增加多种子集成。
它使用与 `bar_gru_15m` 相同的既有 15 分钟派生数据，无需新增数据生成流程。
注意力会额外保留 GRU 输出序列及其加权计算图，资源占用可能高于普通末状态读出；
本次未执行测试、训练或显存基准，未改变 batch_size。

共享输入层的现有边界也沿用：2016 年前不执行 ST 剔除；日频价格未额外复权；
严格分钟窗口要求 16 根 bar、六特征有效，但未检查每根 `minute_count==15`。
这些属于当前公共数据口径，新增 Attention 模型不单独改变它们。

## 时间与产物

首个名义窗口：2013-01 至 2015-09 训练，2015-10 至 2015-12 验证，2016 年预测。
随后每年平移一年，最后一个窗口预测 2026 年至配置截止日 2026-04-24。
实际日期取交易日历；训练末尾清除标签跨入验证期的样本，验证末尾只保留
在重训截止日前到期的标签。因此实际最后一个采样日早于名义月末。

模型仍存于 `data/model_workspace/<factor_id>/artifacts`，因子列仍写入
`data/factors/stock/daily`。新配置需要重新训练相应年度模型；本次代码修改不删除
旧权重或已有因子列，也不启动训练或自动改写历史结果。

## Attention 因子命令与输出

在仓库根目录，使用已安装项目及 PyTorch 的 Python 环境执行：

```powershell
# Local Python 3.8.3; these commands were not executed during development.
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\bar_gru_attention_15m.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\bar_gru_attention_15m.toml
# Combined: factor-run --config ml_alpha\factors\bar_gru_attention_15m.toml
```

模型权重及诊断：`data/model_workspace/bar_gru_attention_15m/artifacts/<window_id>/`。
正式因子列 `bar_gru_attention_15m` 合并写入
`data/factors/stock/daily/<year>/<YYYYMMDD>.parquet`，券商 tag 为 `HTZQ`。
`cache_samples=false`；indexed 模式使用显式模型 workspace 下的任务临时快照，
本机路径位于 D 盘，服务器路径相对其仓库目录；由公共流程结束时清理，不向 C 盘
写缓存。运行时 `device="auto"` 优先使用可用 CUDA。

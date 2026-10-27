# Rawbar RESTCN / Transformer

依据东方证券《多模型学习量价时序特征》（因子选股系列之八十三，
文件日期 2022-06-14，报告内发布日期 2022-06-12），
新增 `rawbar_restcn` 和 `rawbar_transformer` 两个正式端到端因子。
券商 tag 为 `DFZQ`，每个模型最终为每只股票输出一个标量。

## 与现有因子共用的数据和训练口径

完全复用 `rawbar_agru` / `rawbar_lstm` 的 provider：

- 每个样本为截至 T 日的连续 30 个交易日、6 通道
  `open, high, low, close, vwap, volume`，张量形状 `[股票数, 30, 6]`。
- 数据从 `data/chn_stock_data/daily/pv` 和相邻 `adj_factor` 读取。
  历史价格统一复权，除以样本最后一日的复权收盘价；成交量除以最后一日成交量。
  仅保留符合现有 provider 完整性条件的样本。
- 在当前窗口训练集上跨股票、时间步拟合每个通道的截断及标准化参数：
  均值 ±5 个标准差截断，再拟合截断后均值/标准差。
  验证和预测复用保存的参数，不重新拟合。
- `mkt_all` 股票池，剔除 ST、北交所，沿用现有停牌/缺失输入处理。
- 每年重训，过去 36 个自然月，前 33 月训练、后 3 月验证，5 交易日采样。
  训练范围 20130101–20251231，预测范围 20160101–20260424。
- 标签 `future_vwap_return_10d`，`lookahead_days=11`；
  使用公共 runtime 进行训练标签到期、训练/验证隔离和模型启用日期控制。
- 单种子 42，`batch_size=3000`，最多 100 epoch、patience 10、
  Adam 学习率 0.001。训练损失和验证指标沿用公共的负截面 Pearson IC。
- `device=auto`：现有 PyTorch CUDA 可用时使用 GPU。

不需要新增 Rust 派生数据或全量预处理文件。
`cache_samples=false`；输入缓存继续使用 provider 的内存策略。
公共训练流程可能使用配置工作区内的临时 mmap 张量，沿用现有生命周期，
不新增 C 盘缓存路径。

## RESTCN

```text
归一化 rawbar [N,30,6]
  → 3 个局部残差块，64 通道，每块两个 kernel=3 的 Conv1d
  → 4 个 TCN 残差块，dilation=1,2,4,8
  → 最后一个时间步 [N,64]
  → Linear(64,1)
  → rawbar_restcn
```

局部块为 Conv → ReLU → Conv → 残差相加 → ReLU。
TCN 块包含两个 weight-normalized 因果空洞卷积，以及 ReLU、Dropout 和残差旁路。
输入通道不同时使用 1×1 卷积对齐残差。

所有时间卷积都采用左侧补零，保持时间长度不变；因此局部块也不会引入未来时间步。
当前设置的理论感受野为
`1 + 2*(3-1)*(3 + 1+2+4+8) = 73` 个时间步，
覆盖整个 30 日输入，多出的部分为左侧零填充。

参数：

| 参数 | 默认值 |
| --- | --- |
| `residual_blocks` | 3 |
| `hidden_size` | 64 |
| `kernel_size` | 3 |
| `num_layers` | 4，表示 TCN 块数量 |
| `dropout` | 0.1，仅用于 TCN 块 |

## Transformer

```text
归一化 rawbar [N,30,6]
  → 拼接 Time2Vec 的 1 个线性项和 1 个正弦项 [N,30,8]
  → Linear(8,64)
  → 2 个 Transformer Encoder 块
  ├→ 时间维度 Linear(30,1) 汇总全部输出 [N,64]
  └→ 最后时间步输出 [N,64]
  → 拼接 [N,128] → Linear(128,1)
  → rawbar_transformer
```

每个块使用多头自注意力、残差与 LayerNorm、前馈网络、残差与 LayerNorm，
即报告图示的 post-norm 结构；前馈网络使用 ReLU。
注意力的时间范围为同一股票已经观测到的 30 个交易日；
不跨股票做注意力，也不读取 T 日之后的数据。

Time2Vec 使用 `ω₀τ+φ₀` 和 `sin(ω₁τ+φ₁)`。
本项目将窗口内相对位置等距编码为 `[0,1]` 的 τ，并在 checkpoint 中保存位置向量。
报告未明确 τ 对应哪个原始时间字段，因此这是明确的工程选择，
不能将此版本解释为使用了真实年月日或节假日编码。

参数：

| 参数 | 默认值 |
| --- | --- |
| `hidden_size` | 64 |
| `num_layers` | 2，表示 Encoder 块数量 |
| `attention_heads` | 4 |
| `feedforward_size` | 128 |
| `time2vec_periodic_dim` | 1，另有 1 个线性项 |
| `time_coordinate` | `relative_position` |
| `dropout` | 0.1 |

## 与研报完整系统的区别

报告还在 `mschars/l2chars` 上分别训练因子单元，并通过 LightGBM 合成，
最后比较五模型因子层和分数层融合。本次按现有 rawbar 单输出因子的口径，
只新增两个编码器和各自的标量输出头，不接 LightGBM，不创建多模型融合策略。
训练标签、年度滚动方案、单种子和 IC 损失均沿用当前项目设置。

报告披露了底部 64 通道残差结构、TCN 思路、Time2Vec 拼接和时间汇总读出方式；
TCN 层数及 dilation、Transformer 宽度/层数/头数、Dropout、
τ 的相对位置定义和单输出头属于本项目配置。
因此不能将报告的多数据集、多次训练集成回测结果直接归于这两个因子。

## 正式 CLI 与输出

在仓库根目录执行；本轮实现没有自动运行以下命令。

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
$env:PYTHONPATH = "$PWD\ml_alpha"
$env:PATH = "D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;" + $env:PATH

# 分别完成训练和预测。
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_restcn.toml
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_transformer.toml
```

也可以将 `factor-run` 替换成 `factor-train` 或 `factor-predict` 分开执行。
无需 `materialize` 或派生 CLI；需提前备齐训练历史、标签和交易过滤数据。

- 权重：`data/model_workspace/<factor_id>/artifacts/<window_id>/model.pkl`，
  同目录保存 manifest、loss history、model info。
- 日度因子：`data/factors/stock/daily/<year>/<YYYYMMDD>_ml.parquet`，
  分别新增 `rawbar_restcn` 或 `rawbar_transformer` 列。
- 元数据由公共因子流程写入，自动带 `e2e/model_generated`，
  并加入配置中的 `DFZQ` 和模型标签。

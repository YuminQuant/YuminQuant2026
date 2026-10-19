# DFZQ rawbar 单输出因子

依据东方证券《周频量价指增模型》（2022 年 3 月，因子选股系列之八十一）第 6–7 页。
仅实现 rawbar 数据的两个一元因子单元：`rawbar_agru` 和 `rawbar_lstm`。
每支股票每天各输出一个因子值，不生成多元输出、不施加正交惩罚、不接动态加权模型。
配置与注册表标记 DFZQ，运行时元数据还会包含 e2e、model_generated。

## 输入

复用现有日线 `data/chn_stock_data/daily/pv` 和复权因子
`data/chn_stock_data/daily/adj_factor`，无需先派生新数据。
输入为截至 T 日的 30 个交易日 × [open, high, low, close, vwap, volume]。
所有价格先乘对应日期的 adj_factor，再统一除以 T 日复权收盘价，
即 `price(t)*adj(t)/(close(T)*adj(T))`；成交量保留行情原口径并除以 T 日成交量。
VWAP 沿用项目日线单位换算 `amount*10/vol`，通过同步调整 amount 实现价格复权。
缺失复权日文件报错；缺失/非法个股复权、日线或当前日零成交量的窗口不输出。
新 provider 继承 BarPanelProvider，原有日线/分钟 GRU provider 行为保持不变。

## 网络及复现边界

- AGRU：单层 GRU 的每个时间步隐藏状态经过 `Linear→tanh→Linear(1)` 打分，
  沿时间 softmax，加权求和后接 FC(1)。本篇及前序报告可获取正文未披露注意力公式，
  这是明确记录的时间注意力实现选择，不声称逐参数复现研报。
- LSTM：单层 LSTM 最后一层的最后隐藏状态接 FC(1)。
- 两者都是负日期截面 Pearson IC，直接输出一个分数；不使用多元单元的 BN/正交损失。

前序报告参考：2021-03-11《神经网络日频 alpha 模型初步实践》，
公开转载 https://www.scribd.com/document/549815616 。

研报强调跨日期和股票统一的面板预处理。本实现每个年度窗口仅用训练样本，
按六个字段分别汇总所有股票和 30 个时间步，先估计均值/标准差，
按均值 ± 5 标准差截断，再估计截断后的均值/标准差做标准化。
分块两遍估计，不复制整套转换后 tensor；统计量注册为网络 buffer，随权重保存，
验证/预测不重新估计，预测结果不依赖同批其他股票。
截断公式和每年重新估计属于工程选择；原报告未给出具体极值公式，使用测试期前固定参数。
公共日截面变换设为 none，避免误加逐日 z-score；标签保持原收益值，不影响 Pearson IC。

## 统一项目参数

- 年度重训，近 36 月中前 33 月训练、后 3 月验证，每 5 个交易日采样。
- 首个名义窗口：2013-01～2015-09 训练，2015-10～2015-12 验证，2016 年预测。
- 10 日 VWAP 标签、T+11 到期隔离；不同于研报的 10 日收盘收益标签。
- 沿用 mkt_all、ST/BJ 过滤；不是严格复现研报中证全指股票池。
- 单种子 42、隐藏维度 30、单层、Adam lr=0.001、100 轮上限、patience=10、batch_size=3000。
  这些具体超参数沿用现有项目基础方案，未采用研报 5 次独立训练集成。
- 输入 30 日与训练窗口 36 月是两个不同参数，不能混为一谈。

训练、GPU/CPU 选择、分批验证、预测、保存/加载和诊断继承 BarGRU 公共训练实现。
只在模型构建和训练统计量拟合处扩展。已有 mmap 临时 tensor 仍在 D 盘项目
`data/model_workspace/<factor>/cache/scratch`，不新增持久预处理缓存或 C 盘缓存。

## CLI

从项目根目录执行：

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
$env:PYTHONPATH = "$PWD\ml_alpha"
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_agru.toml
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_lstm.toml
```

也可拆为 `factor-train`、`factor-predict`。权重位于
`data/model_workspace/rawbar_agru/artifacts` 和 `data/model_workspace/rawbar_lstm/artifacts`；
因子列写入 `data/factors/stock/daily/<year>/<date>.parquet`。

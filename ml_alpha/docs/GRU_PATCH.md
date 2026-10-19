# GRUPatch1 / GRUPatch2

来源：华泰证券 2024-03-14《如何捕捉长时间序列量价数据的规律》，
第 6–8 页 PatchModel1/2、图表 9–12。本次实现其中的 15 分钟原始量价实验，
不包含需要额外逐笔特征的 30 分钟 22 因子实验。

## 输入与模型

两个因子使用已有 `data/derived/stock/bar/15m`，过去 20 个交易日，每天 16 根 bar，
字段顺序为 open/high/low/close/vwap/volume。provider 先逐字段除以完整 320 步序列均值，
公共 dataset 再逐位置做截面 z-score；标签也做截面 z-score。
strict 模式要求完整交易日序列。数据顺序为从旧到新，再按日内时点排列，
模型将 `[N,320,6]` 变为 `[N,20,16,6]`，不重新计算行情或引入新的数据缓存。

- `gru_patch1` / `GRUPatch1AlphaModel`：共享一个单层 GRU(6→30)，
  分别处理每天的 16 步并取末状态；将 stock×day 合并为 batch 进行计算。
  每天一个标量注意力分数 `tanh(w·h+b)`，沿 20 天 softmax 后加权求和，
  再经过 BN(30)、FC(30→1)。研报表格中的“20 个”是共享参数的 20 次应用。
- `gru_patch2` / `GRUPatch2AlphaModel`：六个独立单层 GRU(16→30)，
  每个字段输入 `[N,20,16]`，20 天为时间步、16 个日内时点为特征。
  取各 GRU 末状态，按原字段顺序拼接 180 维，经过 BN(180)、FC(180→1)。

网络实现位于 `yq_ml_alpha/models/gru_patch_model.py`，训练、验证、预测、
保存/加载、诊断复用 `BarGRUAlphaModel`。基类只增加参数规范化和网络构建扩展点，
原 BarGRU 的默认网络与历史权重加载结构不变。Patch 权重记录架构标识，
两个模型不能互相误载。patch 长度和字段数可配置，不写死为 16 和 6。

## 训练口径

遵循研报：负截面 Pearson IC、Adam、lr=0.005、隐藏维度 30、单层、
最多 100 轮、验证集连续 10 轮未改善则早停、10 日收益标签。
成交价格沿用项目 VWAP：`future_vwap_return_10d = VWAP(T+11)/VWAP(T+1)-1`。

沿用用户本地规范：batch_size=3000、单种子 42、device=auto；每年重训，
过去 36 个月的前 33 个月训练、后 3 个月验证，每 5 个交易日采样。
公共层按 T+11 标签到期边界截断并清除跨训练/验证边界的样本。
这覆盖研报 batch_size=5000、半年重训、4:1 切分的设置。
验证分批前向后计算整个日期的 IC，不取分块 IC 的均值。

首个名义窗口为 2013-01 至 2015-09 训练、2015-10 至 2015-12 验证、
2016 年预测；逐年滚动，预测截止 2026-04-24。
实际可用性取决于现有 15 分钟派生数据和标签覆盖，输入需包含首次样本日前的 19 个交易日。

## CLI 与路径

从项目根目录执行，两个因子分别运行：

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
$env:PYTHONPATH = "$PWD\ml_alpha"
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\gru_patch1.toml
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\gru_patch2.toml
```

也可将 `factor-run` 拆为 `factor-train` 和 `factor-predict`。
权重和诊断在 `data/model_workspace/gru_patch1/artifacts`、
`data/model_workspace/gru_patch2/artifacts`，由公共层按年度窗口管理；
因子列 `gru_patch1` / `gru_patch2` 写入 `data/factors/stock/daily` 的每日宽表。
配置和注册表均为 active，无 deprecated 标签。仅代码接入不会生成权重、预测值或数据元数据。

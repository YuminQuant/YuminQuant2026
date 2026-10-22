# Huatai factor-time CNN / 华泰因子时序卷积网络

`models/mdl_000027.toml` implements the architecture in 华泰证券《人工智能选股之
卷积神经网络》（华泰人工智能系列之十五，2019-02-13），pp11–14、图15。
It is a factor-combination model, using the existing formal factor library.
It does not require raw-price images or a new Rust derived-data pipeline.

按用户要求，直接替换 `models/cnn_model.py:CNNAlphaModel` 中原有的单日 Conv1d
实现，保留类路径。原来没有对应的正式编号配置；本次追加 `mdl_000027`，其他编号
保持不变。旧一维 CNN 权重不能复用，加载缺少序列参数的旧 checkpoint 会明确报错。

## Architecture / 结构

每只股票的输入是 F 个正式因子 × 5 个连续交易日。共用 `load_sequence`，历史日期
从旧到新；列先按日期、再按 provider 的因子顺序展开。模型在每个 batch 内恢复为
`[N, 1, F, 5]`，不会把时间轴误当成独立股票或另造一个单日因子维度。

```text
[N, 1, F, 5]
  -> Conv2d(1, 10, kernel=(5,5), stride=1, padding=0)
  -> ReLU -> flatten: 10*(F-4)
  -> Linear(100) -> ReLU -> Dropout(0.8)
  -> Linear(70)  -> ReLU -> Dropout(0.8)
  -> Linear(40)  -> ReLU -> Dropout(0.8)
  -> Linear(1): predicted 10-day VWAP return
```

No pooling. For the report's F=82, the flattened width is 780, matching figure15.
For the local library, F is discovered from eligible formal factors. Factor
order matters to the local convolution: this implementation retains provider
order, recorded by the shared feature manifest. It does not infer economic
factor families or silently rearrange them by correlation. Explicit feature
lists can specify an ordering; changing it requires retraining.

卷积采用 Xavier uniform 初始化。全连接权重采用零均值截断正态分布，标准差
`dense_init_std=0.1`，截断于 ±2 倍标准差，偏置为 0；分布尺度与截断边界是可说明的
工程默认值，报告没有披露。图15标明 ReLU；Dropout 放在三个隐藏全连接层之后。
报告 p13 明确称 80% 为连接断开比例，因此 `dropout=0.8` 表示丢弃 80%，不是保留 80%。
具体 Dropout 位置未在图中展开，这里采用上述可配置的实现。

## Local contract / 本地口径

- 报告原输入为 82 个因子 × 5 个月末截面；这里沿用本地全部有效正式因子及
  连续 5 个交易日。采样每 5 日一次与序列包含连续 5 日是两件事。
- 报告 CNN 使用前/后 30% 的二分类；按已确认的回归优先约定，本地使用标量
  线性输出、原始 10 日 VWAP 收益标签与 MSE，不输出分类概率。
- 每个源日期截面做 MAD5 截尾、z-score、缺失填 0，收益标签不变换；不额外加入
  报告的行业均值填缺和行业市值中性化。历史股票缺失沿用现有序列加载器填 0 规则。
- `mkt_all`，剔除 ST、北交所；训练范围 2016-01-01 至 2025-12-31；年度重训，
  近 36 个月按前 33 月训练、后 3 月验证，每 5 个交易日采样；每日预测范围
  2019-01-01 至 2026-04-24。实际有效窗口取决于数据可用性。
- 公共层仍处理训练/验证和样本外的标签到期边界，`lookahead_days=11`，
  T 日收盘训练时末截面不晚于 T−11。预测不读取未来收益。
- RMSprop、lr=0.001；单种子 42；batch_size=1000；最多 100 epoch；验证 MSE
  连续 10 轮未改善则早停并恢复最优权重。这些训练控制沿用新增基础模型口径。

## Shared implementation / 公共实现

CNN 复用 `MLPAlphaModel` 的分批训练/验证/预测、MSE、早停、诊断和 checkpoint，
通过 `_build_model` 扩展网络构造，不复制训练循环。MLP 新增可选 RMSprop，默认
仍是 Adam，原 MLP 的结构与默认行为不变。公共 runtime 将 CNN 路由到现有序列
加载器，同时拒绝按行拆分验证集。没有新增磁盘缓存或 C 盘写入路径。

`device="auto"` 在可用时使用 CUDA 训练，否则 CPU。完整平铺输入保留在 CPU，
只将当前训练 batch 搬到 GPU；二维变换和卷积也只针对该 batch。预测沿用共享
MLP 的分批 CPU 前向。模型输入内存随 N×F×5 增长，不是只读取当天一个截面。

## CLI and output / 命令与输出

在项目根目录、已配置好依赖及数据的环境运行：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000027.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000027.toml
# Combined: model-run --config ml_alpha\models\mdl_000027.toml
```

服务器采用相同配置，在已准备好的 Python 环境中：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000027.toml
```

权重和诊断在 `data/model_workspace/mdl_000027/artifacts/<window_id>/`。
预测写入 `data/models/<YYYY>/<YYYYMMDD>.parquet` 的 `mdl_000027` float32 列。
配置、代码接入不代表已执行训练或验证研报收益；服务器依赖和数据仍需先行准备。

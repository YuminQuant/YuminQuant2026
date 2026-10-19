# 多元 GRU 日频量价因子

`rawbar_multi_gru` 参考东方证券 2023-06-06《基于循环神经网络的多频率因子挖掘》的
多元输出结构，仅使用 day/rawbar 数据，对外输出一列因子，券商标签为 `DFZQ`。

## 网络及输出

30 日 × 6 维量价 → 两层 GRU（隐藏维度 64，层间 dropout 0.1）→ 最后时点隐藏状态
→ 全连接 64 维 → BatchNorm → 固定权重 1/64 求和。

等权合成在模型 `forward` 最后完成，权重保存为 buffer，不参与优化；没有可学习的最终
加权层，也不在合成后额外映射。预测、验证、保存加载都使用同一合成逻辑。
内部 64 维输出不自动注册为 64 个独立因子。

## 项目口径与研报差异

- 输入复用 rawbar：复权价格相对当日复权收盘价、成交量相对当日成交量，30 日完整序列。
  训练窗口内拟合截尾及标准化参数并随模型保存，验证和预测只复用。
- 训练损失沿用项目的负 Pearson IC，加内部因子截面相关矩阵非对角元素的均方根惩罚，
  默认系数 0.05。这是项目配置，不是研报披露参数；研报采用 BN 后 MSE 与正交性惩罚。
- 相关性惩罚在同一日期的训练 batch 内计算；batch_size 维持 3000，验证用完整日期 IC。
  验证损失不含正则项，提前停止依据验证负 IC。因子 BN 预测时使用训练运行统计量，
  不随预测股票分批方式重新标准化。
- 标签为 10 日 VWAP 收益，沿用公共层 11 个交易日的标签到期隔离；不额外引入研报的
  收盘收益中性化标签。训练前必须保证标签库已派生该标签。
- 年末重训，回看 36 个月，前 33 个月训练、后 3 个月验证，训练每 5 个交易日采样。
  使用单种子 42，100 epoch、patience 10，学习率 0.001。
- 配置训练日期 2013–2025，预测日期 2016-01-01 至 2026-04-24；实际窗口和可用日期
  由公共调度及数据覆盖范围决定。没有新增磁盘预处理缓存。

## 使用

在仓库根目录使用现有 Python 3.8.3 环境：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = "$PWD\ml_alpha"
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_multi_gru.toml
```

也可分别调用 `factor-train` 和 `factor-predict`，传入同一配置。
模型保存在 `data/model_workspace/rawbar_multi_gru/artifacts`；最终因子写入
`data/factors/stock/daily` 的公共因子表，列名 `rawbar_multi_gru`。

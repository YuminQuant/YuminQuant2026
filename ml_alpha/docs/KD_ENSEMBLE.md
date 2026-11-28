# KD-Ensemble

`kd_ensemble` 是东方证券第 108 篇的端到端因子。原始输入为日频量价 week、30 分钟 msnew 和基本面 LFQ。三路 ASTGNN 分别输出 64 个分量，并按交易日和股票取交集，合成学生所需的 192 维输入。

`factor-train` 在同一个任务内训练三路 ASTGNN。分支训练配置随程序发布，年度滚动，实际生效日由公共时间窗规则计算。学生训练以当时已生效的分支预测作为输入。同一次任务中，每路每个年度只训练一次，重叠季度复用已算出的样本外日截面。

XGBoost 教师以训练集原始 10 日 VWAP 收益拟合并生成软目标。MLP 学生用真实收益 MSE 和教师预测 Smooth L1 加权训练，按验证 MSE 选权重。学生窗口沿用近 60 个月季度更新、57 个月训练加 3 个月验证、每 5 个交易日采样、标签到期隔离。首批学生输入自 2016 年起，KD 从 2021 年起每日预测。

每个季度的 `data/model_workspace/kd_ensemble/artifacts/<window>/model.pkl` 包含 MLP 和该窗口预测所需的全部 ASTGNN 权重。预测读取这一个完整模型和三路原始或派生输入。192 个分量只在内存中存在，股票日频 `YYYYMMDD_ml.parquet` 只写一列 `kd_ensemble`。XGBoost 只用于生成训练期软目标。

先准备 30 分钟行情和 LFQ 派生数据，然后运行下列命令。Slurm 训练用一个任务处理选定的季度窗口以复用分支权重。

```powershell
python -B -u -m yq_ml_alpha factor-train --config ml_alpha/factors/kd_ensemble.toml
python -B -u -m yq_ml_alpha factor-predict --config ml_alpha/factors/kd_ensemble.toml
```

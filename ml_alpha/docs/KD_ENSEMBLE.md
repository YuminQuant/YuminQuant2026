# KD-Ensemble

`kd_ensemble` 是东方证券第 108 篇的端到端因子。原始输入为日频量价 week、30 分钟 msnew 和基本面 LFQ。三路 ASTGNN 分别输出 64 个分量，并按交易日和股票取交集，合成学生所需的 192 维输入。

`factor-train` 每年使用一个 60 个月的历史窗口。前 36 个月用于三路 ASTGNN，其中前 33 个月拟合，最后 3 个月用于选权重。三路权重随后冻结，计算后续 21 个月的学生训练分量和最后 3 个月的学生验证分量。分支时序预热也从五年窗口起点开始，所需历史不足时跳过该样本。每 5 个交易日取一个截面，各阶段按标签到期日隔离。

XGBoost 教师以训练集原始 10 日 VWAP 收益拟合并生成软目标。MLP 学生用真实收益 MSE 和教师预测 Smooth L1 加权训练，按验证 MSE 选权重。首个窗口使用 2011–2015 年数据，三路 ASTGNN 使用 2011–2013 年阶段，学生用 2014 年至 2015 年 9 月训练、2015 年第四季度验证，2016 年开始每日预测。

每个年度的 `data/model_workspace/kd_ensemble/artifacts/<window>/model.pkl` 包含 MLP 和该窗口预测所需的全部 ASTGNN 权重。预测读取这一个完整模型和三路原始或派生输入。192 个分量只在内存中存在，股票日频 `YYYYMMDD_ml.parquet` 只写一列 `kd_ensemble`。XGBoost 只用于生成训练期软目标。

先准备 30 分钟行情和 LFQ 派生数据。Slurm 每年度提交三个单 GPU 分支任务，分别使用执行配置的 `training_stage = "astgnn_week"`、`"astgnn_msnew"` 和 `"astgnn_lfq_daily"`，并共用同一个 `training_run_id`。三路作业各自将权重写至该年度模型目录中的 `<分支名>.pkl`。本年度三路均成功后，运行 `training_stage = "distill"`。蒸馏校验三份权重的运行编号、窗口和配置，在内存中计算分量并保存同目录的 `model.pkl`。本地单一命令仍可串行完成四个阶段。

```powershell
python -B -u -m yq_ml_alpha factor-train --config ml_alpha/factors/kd_ensemble.toml
python -B -u -m yq_ml_alpha factor-predict --config ml_alpha/factors/kd_ensemble.toml
```

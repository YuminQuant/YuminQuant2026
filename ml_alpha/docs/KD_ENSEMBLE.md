# KD-Ensemble

`mdl_000041` 是东方证券第 108 篇的项目版因子合成模型。输入为 `astgnn_week`、`astgnn_msnew`、`astgnn_lfq_daily` 各 64 个弱因子，共 192 列。

三个上游的独立 `factor-predict` 只写正式因子列。week 与 msnew 可以沿用已有的 64 分量权重，LFQ 需先按 64 分量配置训练。

KD 通过统一的组件接口在同一进程中调用三路已冻结的上游模型。每批目标日期按各路实际生效的年度权重划分，检查模型清单和训练截止日，然后从原始或已有派生输入推理。三路输出在内存中按交易日和股票对齐，使用三路均可用的股票。分量与中间矩阵均不写盘。季度窗口内教师与学生共用一次组装的训练矩阵。上游网络推理发生在加载训练、验证或预测数据时，学生每个 epoch 直接读取内存矩阵。

XGBoost 教师仅在训练分片拟合原始 10 日 VWAP 收益，并为该分片生成软目标。MLP 学生用真实收益的 MSE 与教师预测的 Smooth L1 加权训练，按验证集真实收益 MSE 选权重。研报未披露软目标构造、Smooth loss 公式或权重，这里取教师回归分数、Smooth L1 和权重 0.5，均衡设置见 TOML。模型保存和推理只使用学生。

KD 使用近 60 个月的滚动窗口，每季重训，前 57 个月训练、后 3 个月验证，5 个交易日固定采样，最长标签在 11 个交易日后到期。首批训练样本从 2016 年开始使用上游样本外分量，KD 从 2021 年开始每日预测。

从仓库根目录执行，先准备 LFQ 派生数据，再按以下顺序运行。

```powershell
python -B -u -m yq_ml_alpha factor-train --config ml_alpha/factors/astgnn_lfq_daily.toml
python -B -u -m yq_ml_alpha model-train --config ml_alpha/models/mdl_000041.toml
python -B -u -m yq_ml_alpha model-predict --config ml_alpha/models/mdl_000041.toml
```

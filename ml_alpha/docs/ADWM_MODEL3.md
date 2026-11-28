# ADWM Model3

`adwm_model3_daily` 实现东方证券第 112 篇研报的 Model3。模型按年重训，最早从 2016 年预测，每个窗口回看 60 个月，每 5 个交易日取一个训练截面。

- 输入为最近 120 个交易日的 56 项时点基本面指标、56 个缺失标记、9 项 CNE6 一级风格暴露。基本面数据复用 `data/derived/stock/astgnn_lfq` 的 `derive-fundamental-lfq` 产物，逐日做 MAD5 缩尾和 z-score。Barra 逐日做 z-score，缺失填 0。原始 SIZE 另在内存中保留作标签中性化，首批样本从具备完整 CNE6 历史后开始。
- 前 36 个月训练 ABCM。一层 64 维 GRU 输出 64 个弱 alpha 和 12 项风险暴露。非对称风险图在完整当日截面上计算，把当日及前两个交易日的图表示按股票键相加后，再投影成风险暴露。某只股票缺失某期历史图时，该期对其无贡献。
- ABCM 对各 alpha 使用 10 日 VWAP 收益 MSE，对等权 alpha 加 RankNet，对完整因子矩阵用 20 日 VWAP 收益的未解释方差比例及非对角相关性惩罚。阶段一按自身训练目标选权重，使用全部成熟训练截面。
- 接着 21 个月冻结 ABCM，将生成的弱 alpha 和风险暴露保留在内存供各轮复用。门控使用风险状态生成带噪 Top-8 Softmax 权重，计算加权 alpha。10 日 VWAP 收益逐日对当时的中信一级行业和 CNE6 SIZE 做联合回归、取残差。缺行业、SIZE 或标签的股票不参与监督。
- 门控使用中性化收益 MSE 和 Router z-loss，最后 3 个月按无噪声 MSE 选权重。两阶段上限均为 100 轮，连续 10 轮无改善早停，单种子 42。阶段边界按各自所需最长标签到期隔离，ABCM 为 21 个交易日，门控为 11 个交易日。

研报未披露的损失系数由 TOML 配置。首版取 RankNet 和 Router z-loss 各 0.001，风险未解释方差和相关惩罚分别为 1.0 和 0.1。

每个窗口只保存一套联合权重 `data/model_workspace/adwm_model3_daily/artifacts/annual_YYYY/model.pkl`。预测用同一权重计算日度弱因子和无噪声门控，最终写入 `data/factors/stock/daily/YYYY/YYYYMMDD_ml.parquet` 的 `adwm_model3_daily` 一列。

```powershell
& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\adwm_model3_daily.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\adwm_model3_daily.toml
```

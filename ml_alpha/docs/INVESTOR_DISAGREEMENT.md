# Investor recognition and disagreement / 投资者认可度与分歧

`mdl_000028` implements the tree-model ensemble from 华泰证券《机器学习模拟投资者
分歧》（人工智能系列80，2024-06-15），p7 图9及 pp13–14 的认可度/复合因子。
This is a factor-combination model. Existing formal factors supply its input;
it does not need raw-bar sequences, Rust derived data, analyst coverage or fund
holdings. Those last two subjects appear in the report's comparisons/economic
analysis, not as required inputs to the main ensemble.

## Model and score / 模型及分数

用户确认的本地版本：50 个 LightGBM 回归器，每个使用约一半有效正式因子。
设输入因子总数为 F，每位“投资者”使用 `ceil(0.5 * F)` 个因子，单个子集内不重复。
不同投资者的子集允许交叠，独立抽样，不根据验证集或预测收益挑选子集。
若 F 太小而无法产生严格子集，直接报错，不退化为所有投资者使用全部因子。

每个训练窗口用 seed=42 的固定随机数生成器生成 50 个子集；输入因子目录及顺序
相同时，每年使用相同子集。模型随机种子依次为 42–91。保存时同时保存每个子模型、
列索引、列名及随机种子；预测加载已有子集，不重新抽样。若因子目录发生变化，
需按公共 manifest 规则重新训练，不将旧列索引套到新列顺序。

每只股票 i、每个日期 t 有 50 个预测值 p(k,i,t)：

```text
recognition(i,t) = mean_k p(k,i,t)
disagreement(i,t) = sqrt(mean_k (p(k,i,t) - recognition(i,t))^2)
score(i,t) = 0.5 * zscore_t(recognition(i,t))
           + 0.5 * zscore_t(-disagreement(i,t))
```

认可度越高、分歧越低，最终分数越高。它是标准化后的复合选股分数，**不是预测收益率**。
标准差使用总体口径 ddof=0。该选择在固定 50 个完整预测下，不影响随后截面标准化的
负分歧分量。报告写明等权合成，但未披露标准化细节；这里采用用户确认的分量分别
按日期 z-score 后等权相加，不直接把均值和负标准差按原始尺度相加。

标准化调用公共 `apply_cross_section_transform(..., "zscore")`，按每个日期的完整
合格股票截面计算，零方差分量记为 0。内部股票批次只用于预测内存控制，**不在每个
批次分别做 z-score**，也不把不同日期混在一起计算均值/标准差。预测不读取标签。
外部调用方必须与项目 pipeline 一样，传入完整日期截面，而非任意部分股票。

`predict_components()` 可供诊断取得原始认可度和负分歧；正式 `predict()` 只输出
一列复合分数。不会额外向正式因子库或模型表写入 50 列中间预测。

## Training / 训练

- 本地全部有效正式日因子，沿用自动发现及 deprecated/model-generated 排除规则。
- 每个日期输入做 MAD5 截尾和 z-score，缺失值填 0；10 日 VWAP 收益标签保留原尺度。
- `mkt_all`，剔除 ST、北交所，不额外剔除涨跌停。
- 训练日期范围 2016-01-01 至 2025-12-31；年度重训，过去 36 个月按前 33 月训练、
  后 3 月验证，两段分别每 5 个交易日采样；每日预测配置范围为
  2019-01-01 至 2026-04-24，实际窗口受数据可用性限制。
- 公共层 `lookahead_days=11` 处理训练/验证隔离和样本外标签到期边界。
  T 日收盘训练时最后可用截面不晚于 T−11，模型不另写日期切分逻辑。
- 50 个模型使用同一训练分区、同一目标，只改变因子子集和记录的随机种子；
  不使用额外行 bootstrap。验证分区不参与拟合，不进行 Optuna、子集筛选或集成权重拟合。
- 验证只在整套模型完成后执行一次预测，报告认可度 MSE 和复合分数平均日 IC，
  不将复合分数与原始收益计算 MSE，也不进行逐树验证。

研报使用 43 个因子、每个模型随机 20 个、近 6 年按 7:3 划分。用户选择改为本地
因子库、每个约一半因子以及上述项目窗口。报告没有公开全部因子清单和软件版本，
本实现不宣称逐项复现其输入或收益结果。早期多深度模型的实验分支和基金持仓/近期
收益的双重排序分析不在本配置中。

每个 LightGBM 使用固定默认风格参数：100 棵树、31 叶、深度不限制、学习率 0.1、
叶子至少 20 个样本、无额外行列采样、L1/L2 均为 0。故每个年度窗口最多训练
50×100=5,000 棵树，不是 Optuna 运行 50 次；遇到无法继续分裂时实际树数可能减少。
这些值显式写入配置以避免库升级改变默认行为，不是经本地优化的最优超参数。

`device_type="gpu"` 使用 LightGBM OpenCL GPU 训练，max_bin=255，CPU 辅助线程为 8。
未配置 GPU/OpenCL 支持时由 LightGBM 报错退出，没有自动 CPU 重试或跳过部分投资者
继续计算的路径。LightGBM 预测沿用其 CPU 推理接口。模型本身没有安装或探测 GPU 的副作用。

## Resources and artifacts / 资源与产物

训练矩阵仅构建一次，50 位投资者顺序训练，内存中每次只增加当前子集矩阵；全部
训练后的子模型权重需要保留。共享完整矩阵 float32 的元素存储约为 4×N×F 字节，
当前半数因子切片约为 4×N×ceil(F/2) 字节；此外还有 DataFrame、清洗转换的临时
副本、LightGBM 分箱和 GPU 工作内存，实际峰值高于这两个数之和。

预测每批最多 10,000 行，整批特征转换一次，依次调用 50 个子模型，用 float64 的
Welford 算法累计均值/方差。不会构造 N×50 的全量预测表。所有股票的两个最终分量
保留到完整截面标准化结束。模型数量增加会线性增加训练和推理工作量；此处没有训练
或性能实测结论。

- `data/model_workspace/mdl_000028/artifacts/<window_id>/model.pkl`：整个集成及子集映射。
- 同目录 `model.manifest.json`：公共配置、窗口、特征指纹。
- `model_info.json`：实际参数、LightGBM 版本、每位投资者列名/种子、验证指标。
- `investor_history.parquet`：每位投资者的编号、种子、特征数和拟合耗时。
- `data/models/<YYYY>/<YYYYMMDD>.parquet`：`trade_date`、`ts_code` 键下的
  `mdl_000028` float32 分数列。

`cache_samples=false`，不新增磁盘预处理缓存，不写 C 盘缓存。50 个子模型作为一个
model artifact 保存，仍使用公共 model-train/model-predict 工作流。

## CLI

本地在仓库根目录运行，依赖和数据必须已经准备好：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000028.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000028.toml
# Combined: model-run --config ml_alpha\models\mdl_000028.toml
```

服务器在具备 GPU LightGBM 及项目数据的环境中使用相同配置：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000028.toml
```

本次代码接入不安装服务器依赖、不生成数据、不执行测试或训练。

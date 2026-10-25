# ML Alpha / 机器学习 Alpha

`ml_alpha` 使用 Python 机器学习框架训练和推理模型信号、端到端正式因子。支持因子表、bar 序列、派生特征及部分图/文本特征；具体输入和训练窗口由 TOML 配置决定。

`ml_alpha` trains and serves model signals and end-to-end formal factors using Python ML frameworks. Inputs include factor tables, bar sequences, derived features and selected graph/text features. TOML configurations define inputs and training windows.

## 配置层次 / Configuration layers

| 目录 / Directory | 用途 / Purpose |
|---|---|
| `models/*.toml` | 模型实验与组合信号 / Model experiments and combined signals |
| `factors/*.toml` | 注册为正式因子的端到端模型 / End-to-end models registered as formal factors |
| `model_registry.toml` | 模型标识与配置目录 / Model identifiers and configurations |
| `docs/` | 模型结构、研究口径和开发规范 / Architectures, methodology and development contracts |

正式因子的文件名、factor_id 和输出列使用一致的语义名称；模型编号不等同于正式因子名。支持传统回归/分类、树模型、序列神经网络及图模型，是否可运行取决于对应依赖和数据。

Formal factor filenames, IDs and output columns use matching semantic names. Model IDs are separate from formal factor names. Implementations cover regression/classification, tree ensembles, sequence networks and graph models; each requires its own dependencies and data.

`mdl_000033` / `mdl_000034` 为华泰多任务因子合成 MLP，分别采用 UW / DWA 损失加权，
共享网络同时预测 10/20 日收益排序，再等权输出单分数。
结构、双标签到期隔离和 CLI 见 [Multi-task learning / 多任务学习](docs/HUATAI_MULTITASK.md)。

华泰多频率 GRU 系列包含 `bar_gru_15m`、`bar_gru_attention_15m`、
`multi_bar_gru_daily_15m` 和 `residual_multi_bar_gru`；输入、网络、统一训练口径
及命令见 [研报对齐说明 / Report alignment](docs/BAR_GRU_REPORT_ALIGNMENT.md)。

## 安装 / Installation

从仓库根目录安装，按所选配置选择 extras。GPU/CUDA/OpenCL 与模型库版本需要匹配，不要求所有机器采用同一个个人环境。

Install from the repository root and choose extras for the selected configuration. GPU/CUDA/OpenCL and library versions must be compatible; no particular personal machine environment is required.

```bash
python -m pip install -e "./ml_alpha"
python -m pip install -e "./ml_alpha[ml,xgb,torch,optuna]"
```

第二条会安装较多可选依赖，可仅选所需组合。安装后从仓库根目录运行以下示例，并检查 TOML 中的数据路径、训练/验证/预测日期和设备设置。

The second command installs several optional dependencies; select only what you need. Run examples from the repository root after checking data paths, train/validation/prediction dates and device settings in the TOML.

## 常用命令 / Common commands

```bash
python -m yq_ml_alpha --help
python scripts/yq.py model-train --config ml_alpha/models/mdl_000001.toml
python scripts/yq.py model-predict --config ml_alpha/models/mdl_000001.toml
python scripts/yq.py factor-train --config ml_alpha/factors/bar_gru_15m.toml
python scripts/train_models.py factor-train --config ml_alpha/factors/bar_gru_15m.toml --dry-run
python scripts/yq.py factor-predict --config ml_alpha/factors/bar_gru_15m.toml
python -m yq_ml_alpha factor-metadata
```

`model-run` / `factor-run` 连续执行训练与预测；`model-materialize` / `factor-materialize` 单独准备特征缓存。不要批量盲跑示例配置：不同配置可能依赖尚未生成的派生数据或标签。

`model-run` / `factor-run` combine training and inference. Materialization commands prepare feature caches separately. Do not run all example configurations blindly: some require derived inputs or labels that have not been generated.

刷新 Rust 与 ML metadata 可使用 `factor-metadata-all`；ML 因子保留独立 metadata 来源，不应手工修改合并表来注册模型。
Use `factor-metadata-all` to refresh Rust and ML metadata. ML factors maintain their own metadata source; do not manually register models by editing the merged table.

## 输出与复用 / Outputs and reuse

多模型训练支持按原始 refit 窗口独立调度；联合预测支持不同窗口，兼容输入共享、
有界分数缓冲及每日多列一次写出。本地默认串行，可通过独立执行 TOML 开启并发。
详见[多模型执行](docs/MULTI_MODEL_EXECUTION.md)，执行示例为 `execution.example.toml`。

Multi-model fits can run as independent original refit windows. Grouped prediction
supports different windows, compatible input sharing, bounded score buffers and
one multi-column daily write. Local defaults are serial; a separate execution TOML
controls concurrency. See [multi-model execution](docs/MULTI_MODEL_EXECUTION.md).

统一入口在正式 ML 因子生产前刷新一次 metadata；原模块命令保留。普通训练入口默认单任务；scripts/train_models.py 按配置中的模型窗口拆成独立任务，本机顺序执行，部署调度器可并发。带 validation_lookback 的窗口支持年度、半年度、季度、月度更新；窗口 ID、标签清洗和模型路径保持一致。远程张量受预算限制，不在节点写 memmap。详见[统一计算与存储](../factor_engine/docs/ARTIFACT_STORAGE.md)。
The unified entrypoint refreshes metadata once before ML factor production; module commands remain available. Ordinary training remains single-task. scripts/train_models.py creates one task per configured model window, sequential locally or concurrent through deployment scheduling. Calendar windows with validation_lookback support annual, semiannual, quarterly and monthly refits while retaining window IDs, label purging and artifact paths. Remote tensors stay within a RAM budget, not node memmaps. See [execution and storage](../factor_engine/docs/ARTIFACT_STORAGE.md).

- 模型信号通常写入 `data/models/{year}/{date}.parquet`；正式因子写入 `data/factors/stock/daily/{year}/{date}.parquet`。
  Model signals normally go to `data/models`; formal factors go to the daily factor library.
- 模型、manifest 和特征缓存由配置的 `data/model_workspace/` 子目录管理。预测复用已训练模型，并检查配置/特征身份。
  Model artifacts, manifests and feature caches live in configured workspace directories. Inference reuses trained artifacts with identity checks.
- 避免独立预测进程同时修改同一批日文件；先确认写入协调机制。标准入口默认本地存储。
  Avoid independent prediction processes writing the same daily files without coordination. Standard entrypoints use local storage.

## 研究与资源边界 / Research and resource boundaries

Parquet 输入通过统一存储接口直接读取，本机与远程使用相同的按列读取逻辑。日表的 schema 和数据共用一次受控打开；可选缺失列仍填空，IO 或损坏文件错误不会退回全列读取。进程内 LRU 仅缓存原始 Arrow 表和 footer，不缓存预处理结果或模型输出，也不写节点磁盘缓存。

Parquet inputs use the same projected reader through the storage backend locally and remotely. Daily schema checks and data reads share one managed file handle. Optional missing columns remain null; IO/corruption errors never trigger a full-column fallback. The process-local LRU stores only source Arrow tables and footers, not transformed samples or predictions, and never spills to node disk.

`YQ_PARQUET_CACHE_BYTES` 默认 64 MiB，`YQ_PARQUET_METADATA_BYTES` 默认 8 MiB，设为 `0` 可关闭对应缓存。超大表正常读取但不留在缓存中；每次访问检查文件身份，写入、替换、删除会使缓存失效，DatasetBuilder 关闭时释放缓存。预算是额外的读取缓存上限，不是训练张量、已有特征窗口缓存和模型的总内存上限。

`YQ_PARQUET_CACHE_BYTES` defaults to 64 MiB and `YQ_PARQUET_METADATA_BYTES` to 8 MiB; set either to `0` to disable that cache. Oversized tables are read without retention. File identity is checked on access; writes, replacement and deletion invalidate entries, and closing DatasetBuilder clears them. These budgets cover extra read caching, not the combined RAM of training tensors, existing feature-window caches and models.

训练与验证必须遵守标签成熟时间；固定训练、滚动训练及验证划分不是全项目统一常量，以配置和专题说明为准。按行划分验证集不保证时间隔离，不能把它当作严格样本外检验。

Training and validation must respect label maturity. Fixed/rolling training and validation splits are configuration-specific. Row-based validation does not guarantee temporal separation and is not a strict out-of-sample test.

先小范围物化和训练，再调整 batch、缓存与设备参数；更新特征数值口径时应重新生成一致的数据并重训，不能混用旧模型和新特征。

Start with small materialization/training runs before tuning batches, caches and devices. Changed feature definitions require consistent regeneration and retraining rather than mixing old models with new inputs.

## 专题文档 / Guides

- [开发流程 / Development workflow](docs/DEVELOPMENT_README.md) · [性能工作流 / Performance workflow](docs/PERFORMANCE_WORKFLOW.md)
- [传统模型基线 / Classical baselines](docs/HUATAI_BASELINES.md)
- [Bar GRU](docs/BAR_GRU_REPORT_ALIGNMENT.md) · [Raw-bar models](docs/RAWBAR_ALPHA.md) · [ResNet-GRU](docs/RESNET_GRU.md)
- [多变量 GRU / Multivariate GRU](docs/MULTIVARIATE_GRU.md) · [RESTCN / Transformer](docs/RAWBAR_TEMPORAL.md)
- [FactorVAE](docs/FACTOR_VAE.md) · [图与序列模型 / Graph and temporal models](docs/GRAPH_TEMPORAL.md)
- [FactorGCL](docs/DFQ_FACTOR_GCL.md) · [TimesNet](docs/DFQ_TIMESNET.md) · [Neural ODE](docs/DFQ_NEURAL_ODE.md)
- [研报情绪 / Report sentiment](docs/REPORT_SENTIMENT.md)
- [回测与分析 / Backtest analysis](../analysis_tools/README.md)

验证脚本、测试数据和本机安装记录不属于 README 或版本库内容。
Keep validation scripts, fixtures and machine-specific installation logs out of README overviews and version control.

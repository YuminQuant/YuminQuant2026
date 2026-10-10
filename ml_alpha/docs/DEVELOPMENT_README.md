# ML Alpha Development Workflow / 开发工作流与交接

This guide describes the current implementation, not a proposed architecture.
Start here when taking over `ml_alpha`; consult the linked code before changing
behavior. All commands below assume the repository root as the working directory.

本文描述当前实现，不是待实施方案。接手 `ml_alpha` 时先阅读本文，再核对对应代码；
下方命令统一在仓库根目录执行。不要把示例中的长日期范围当作开发验证范围。

## 1. Ownership And Storage / 职责与存储

| Layer / 层 | Location / 位置 | Responsibility / 职责 |
| --- | --- | --- |
| Source / 原始数据 | `data/stock_data/...` | Input only; do not modify during tests / 测试不修改源数据 |
| Derived / 可复用派生数据 | `data/derived/stock/...` | Rust production, daily Parquet / Rust 生产，日度落盘 |
| Labels / 标签 | `data/label/stock/daily` | Forward-return labels; not prediction inputs / 未来收益，不可作为预测特征 |
| Experiments / 模型实验 | `ml_alpha/models/*.toml` | Model-layer configuration / 模型层配置 |
| Formal factors / 正式因子 | `ml_alpha/factors/*.toml` | One semantic factor identity per config / 每份配置对应一个语义化因子 |
| Artifacts / 模型产物 | `data/model_workspace/<id>/artifacts` | Weights, manifests and model diagnostics / 权重、校验清单和诊断 |
| Temporary samples / 临时样本 | `data/model_workspace/<id>/cache` | Optional sample cache, not canonical features / 可选样本缓存，不是正式派生特征 |
| Model signals / 实验信号 | `data/models` | Model-layer predictions / 模型层预测输出 |
| Formal output / 正式输出 | `data/factors/stock/daily/<year>/<date>.parquet` | Final factor columns merged by shared writer / 公共写入器合并最终因子列 |

`model_workspace` is not limited to weights, but reusable derived features must
not be hidden there. In particular, volume logsignatures belong to
`data/derived/stock/logsig_v/<year>/<date>.parquet`; only the final neutralized
`logsig_alpha_v` score belongs to the formal factor library.

`model_workspace` 不只放权重，也可放诊断和临时缓存，但可复用的派生特征不能放在这里。
logsignature 的226维输入在 `derived/stock/logsig_v`，中性化后的最终单列才写入因子库。

## 2. Execution Flow / 执行流程

```text
Raw/PIT data -> reusable derived features (Rust, daily files)
            -> feature provider -> universe/filters -> labels for training only
            -> leakage-safe training windows -> fit/validate -> model artifacts
            -> date-batched prediction -> model-owned postprocessing
            -> shared output writer -> metadata -> separate backtest
```

中文：先准备源数据、派生特征和标签；Python provider 按日期读特征，执行股票池过滤、
训练标签对齐与预处理；公共 runtime 构建训练窗口，训练后保存模型与 manifest；
预测分日期批读取特征，模型返回分数，公共写入器落盘。回测是独立步骤，不包含在训练中。

English: Rust owns reusable derived-data production. Python/PyTorch owns model
training and inference. `factor-run` trains and predicts; `factor-train` and
`factor-predict` separate those stages. Prediction loads validated artifacts.
No command should silently regenerate missing canonical logsignature features.

### Current Logsignature Workflow / 当前 Logsignature 命令

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"

# Produce derived inputs, including dates needed by training and validation.
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-logsig `
  --asset stock --start-date 20110101 --end-date 20260424 --threads 2

# Train and predict separately (factor-run is the combined alternative).
& $py -m yq_ml_alpha factor-train --config ml_alpha\factors\logsig_alpha_v.toml
& $py -m yq_ml_alpha factor-predict --config ml_alpha\factors\logsig_alpha_v.toml

# Merge Python factor metadata, or refresh Rust then Python metadata.
& $py -m yq_ml_alpha factor-metadata --config ml_alpha\factors\logsig_alpha_v.toml
# & $py -m yq_ml_alpha factor-metadata-all
```

中文：沿用已有 Python 3.8.3 GPU 环境，不默认新建 `.venv`，不依赖高版本 base Python。
Rust 派生 CLI 不需要 Python 扩展；但当前 logsignature 模型的最终中性化仍需要
`yq_factor_engine_py`。扩展源码在 `factor_engine/python/yq_factor_engine_py`，
更改扩展接口后需为同一 Python 环境重新构建安装，不能只更新 `.exe`。
不要使用 `KMP_DUPLICATE_LIB_OK=TRUE` 掩盖 OpenMP 运行库冲突。

English: use the existing Python 3.8.3 GPU environment. Do not create a new venv
or switch interpreters implicitly. The standalone Rust derived CLI does not
need a Python extension; model-owned Rust neutralization does. An updated Rust
executable does not update the installed `yq_factor_engine_py` extension. Resolve
OpenMP conflicts through environment consistency, not the unsafe duplicate-runtime flag.

## 3. Files To Add Or Change / 需要编写哪些配置

| Task / 任务 | Files / 文件 |
| --- | --- |
| Experiment / 模型实验 | `models/<model_id>.toml`, catalog entry in `model_registry.toml` |
| Formal factor / 正式因子 | `factors/<factor_id>.toml`, catalog entry in `factor_registry.toml` |
| New model / 新模型 | `yq_ml_alpha/models/<name>_model.py`, tests, TOML `model.class` |
| New input layout / 新输入布局 | `features/<provider>.py`, `data/dataset.py:make_feature_provider`, tests |
| Reusable derived input / 可复用派生输入 | `factor_engine/src/derive`, CLI dispatch in `src/main.rs`, tests and docs |

Registries are catalogs, not substitutes for TOML configuration or generated
factor metadata. Read `pipelines/factor.py` and `output/factor_metadata.py` before
changing active/deprecated behavior. Formal config filename, `factor_id` and
`output.id` must match; do not introduce `mdl_*` or `e2e_fct_*` formal factor IDs.

registry 是目录说明，不能代替 TOML 配置和生成后的 metadata。正式因子的文件名、
`factor_id`、`output.id` 必须一致；弃用行为应核对实现，不能只改 registry 状态。
模型实验沿用模型层标识与 signal 输出，不直接向正式因子库写实验列。

### Formal Config Example / 正式因子配置示例

This is the fixed logsignature workflow, not a generic template for every model.
Copy the nearest existing model config for other architectures. Parameters under
`model.params` are model-specific; unsupported parameters do not acquire behavior
merely because they appear in TOML.

以下以 logsignature 为例；其他模型应复制最接近的现有配置。`model.params` 的含义
由模型实现决定，不能只增加 TOML 参数就假设框架会执行它。

```toml
factor_id = "logsig_alpha_v"
data_root = "data"
data_version = "unversioned"

[output]
kind = "factor"
id = "logsig_alpha_v"
root = "data/factors"
asset = "stock"
frequency = "daily"
base_root = "data/stock_data/daily/pv"
write_workers = 4

[dates]
train = [20110101, 20150930]
valid = [20151001, 20151231]
predict = [20160101, 20260424]

[sample]
train_frequency = "5"
predict_frequency = "daily"

[train_scheme]
type = "static"

[label]
id = "future_vwap_return_5d"

[universe]
id = "mkt_all"

[filters]
exclude_limit = false
exclude_st = true
exclude_bj = true

[preprocess]
cross_section_transform = "none"
feature_fill_value = 0.0

[features]
type = "derived_logsig"
root = "data/derived/stock/logsig_v"
columns = "__all__"

[materialize]
cache_samples = false
cache_dir = "data/model_workspace/logsig_alpha_v/cache"
predict_batch_size = 20

[diagnostics]
enabled = true
print_epoch = true
write_loss_history = true
write_model_info = true
write_window_summary = true

[model]
name = "logsig_orthogonal_mlp"
class = "yq_ml_alpha.models.logsig_orthogonal_mlp_model.LogsigOrthogonalMLPAlphaModel"
artifact_dir = "data/model_workspace/logsig_alpha_v/artifacts"

[model.params]
base_factors = 8
orthogonal_lambda = 0.05
hidden_layers = [64]
dropout = 0.1
epochs = 100
batch_size = 5000
lr = 0.001
weight_decay = 0.0
seed = 42
device = "auto"
patience = 10
neutralize = "barra:SIZE+sector"
```

中文：配置内相对路径由 `config.py:_project_path` 解析，并非相对 TOML 所在目录；
参考现有配置的 `data/...` 写法。CLI 的 `--config` 路径则相对启动目录。
`static` 固定训练；滚动训练参考 `models/mdl_000001.toml` 与 `runtime.build_windows`，
例如 `type="rolling"`、`refit_frequency="monthly_end"`、`train_sample_count=36`。
不要将36个采样日误解为36个交易日；月末采样时它们对应36个月末截面。

English: config-relative data paths are resolved by `config.py:_project_path`,
not relative to the TOML directory. The CLI config argument itself is relative
to the working directory. For rolling models, inspect `mdl_000001.toml` and
`build_windows`; sample counts count sampled dates, not individual stock rows.

## 4. Derived Data Contract / 派生训练数据写法

中文：复用 `DeriveEngine`、request/report、`MarketDataLoader` 和公共派生写入器。
按交易日读取所需列，不读取无关 OHLCV；跨日算法使用有界状态，以股票代码对齐，
不得依赖上一批股票行号。预热只用于计算，不输出请求区间外的数据。
按日写 Snappy Parquet，保留 `trade_date`、`ts_code` 和确定的特征 schema；
空结果也保留 schema，缺失日不跳过补旧数据。明确覆盖/增量、缺失值和单位口径。
写入先完成临时文件再替换目标，失败不能截断已有产物；测试必须使用隔离目录。

English: follow the existing derive engine, request/report, loader and shared
Snappy writer. Project required columns; maintain bounded, instrument-aligned
state; keep warmup out of output. Write stable daily schemas, including empty
results. Define missing-data, units and overwrite semantics explicitly. Use
atomic replacement and isolated test paths. Avoid model-specific copies of the
same reusable feature dataset or accumulating all dates before writing.

Current `derive-logsig` specifics / 当前实现：

- Fixed 5m / 20 trading days / order 10, 226 float32 columns / 固定口径，226列。
- Input columns: `ts_code`, `trade_time`, `vol`; sessions exclude 09:30 / 仅读三列。
- All 48 finite slots are needed on all20 dates; partial five-minute groups are summed / 20日每日期内48槽必须有效，槽内允许分钟不齐。
- `--threads` controls stocks within one date; no `date-batch-size` / 日期串行，股票并行。
- Default overwrite is true; false skips existing files without freshness checks / 默认覆盖；跳过模式不校验源数据修订。
- After source revisions, regenerate affected windows and review `data_version` / 源数据修订后重算受影响窗口，评估模型数据版本。
- `derived_logsig` provider reads files only, fails on missing files/columns / 模型端只读，缺文件或列报错。
- `factor-materialize` rejects this provider; it is not the derived production CLI / 不再通过 Python materialize 生产此数据。

The Rust kernel caches fixed order-10 structures and reuses numerical buffers.
Descending-degree in-place updates preserve the original calculation results.
Do not replace full-path computation with inverse-based rolling removal without
new numerical validation: high-order cancellation can change the features.

Rust 内核缓存固定10阶结构、复用数值缓冲，并通过倒序原地更新减少复制。
不要未经数值验证就改成 signature 逆运算移除历史窗口；高阶抵消误差可能改变特征。

Precision: minute values are promoted to `f64` for aggregation and tensor
arithmetic, then the 226 features are stored as `f32`. This is unchanged by
the kernel optimizations; storage dtype is not intermediate computation dtype.
精度约定：分钟值提升为 `f64` 进行聚合和张量计算，226维特征仍以 `f32` 落盘。
这不是此次优化引入的变化；存储精度与中间计算精度应分别判断。

Other `materialize` paths may write debug samples or legacy provider caches;
they are not automatically streaming or a replacement for Rust derived production.
Inspect the selected provider rather than generalizing from logsignature.

其他 materialize 分支可能生成调试样本或旧 provider 缓存，并不自动具备逐日低内存语义；
接手时应核对具体 provider，不能将 logsignature 的保证推广到所有模型。

## 5. Model Interface And Postprocessing / 模型接口与后处理

Read [AlphaModel / ModelContext](../yq_ml_alpha/models/base.py),
[runtime](../yq_ml_alpha/pipelines/runtime.py), and the nearest model implementation.
`model.class` is dynamically imported; its class receives model parameters from
the runtime. A tabular model implements:

```python
def fit(self, train_data, valid_data, context): ...
def predict(self, data, context): ...  # pd.Series, same row count/order/index
def save(self, path): ...
@classmethod
def load(cls, path): ...
```

中文：读取 `context.feature_columns` 和 `context.label_column`，不要硬编码训练表列顺序。
训练统计量只能在训练集拟合，并随模型保存；验证与预测复用。预测必须保持输入行对齐，
不得把 `trade_date`、`ts_code` 或未来标签误当特征。模型应保存预处理统计、权重及必要参数，
提供可恢复的 load；只加载可信模型文件。新模型的随机种子、CPU/GPU行为和缺失值策略应明确。

English: fit preprocessing statistics on training only and persist them with
weights. Return scores aligned to input rows; never include keys or future labels
as features. Make seed/device/missing-value behavior explicit. Load only trusted
artifacts. The inherited pickle implementation is a default, not a requirement
for torch model storage.

Sequence models may implement `fit_bundle` / `predict_bundle`; see the GRU
implementations. `DatasetBundle` separates keys/labels from float32 sequence
tensors; `TensorSpool` supports mmap storage. Close bundles and release accelerator
memory using the existing lifecycle. Do not expand tensors back into huge pandas
wide tables or assume all tabular training is out-of-core.

序列模型可实现 `fit_bundle/predict_bundle`；复用 tensor/mmap 通路，及时关闭 bundle。
不要为方便把序列重新展开为巨型 pandas 宽表。普通表格模型仍可能将完整训练/验证样本
装入内存；派生数据逐日落盘并不等于训练阶段完全流式。

For logsignature, prediction does model inference/composition, datewise zscore,
then calls Rust `neutralize_daily` through `yq_factor_engine_py` for industry +
Barra SIZE residuals. This is model-owned behavior, not universal preprocessing.
The 226 derived features are not individually neutralized. Do not silently add a
second neutralization in the writer or use inference chunks as regression universes.

logsignature 的预测得分先按日 zscore，再由模型调用 Rust 做行业与 SIZE 中性化。
不是对226个输入分别中性化，也不是所有模型自动具备该步骤。不要在写入器重复中性化；
神经网络推理 batch 不能切断当日完整横截面的统计与回归。

## 6. Safety And Performance / 安全与性能约定

中文：沿用 `runtime.build_windows` 的标签到期检查；自定义标签填写 `label.lookahead_days`。
训练、验证、预测边界不能有未来信息，财务/分析师特征必须保证 PIT。
复用 artifact manifest 校验；变更特征顺序、训练日期或预处理后不要强行加载旧模型。
manifest 不哈希整个数据库；源数据重建时主动更新/确认 `data_version` 并重新训练。
`--resume` 是续跑机制，不是对数据修订的自动检测。

English: preserve label settlement/purge checks and PIT constraints. Supply a
lookahead for custom labels. Keep artifact-manifest validation; never bypass a
mismatch to reuse an incompatible model. Database contents are not fully hashed;
review `data_version` after rebuilding inputs. Resume is not automatic freshness detection.

中文：不同 batch 参数不要混用：`derive-bar.date-batch-size` 控制并发日期；
`derive-logsig.threads` 控制日内股票计算；`materialize.predict_batch_size` 控制预测日期批；
`model.params.batch_size` 是模型自己的训练/推理参数。优先投影读取、日度释放和紧凑数组，
再考虑并行，避免多层线程过量并发。正式输出复用 `DailyWideWriter`，不得自写整库覆盖逻辑。

English: distinguish concurrent-date batches, per-date stock workers, prediction
date batches and model minibatches. Optimize projection and lifetime before
parallelism. Avoid nested oversubscription. Reuse `DailyWideWriter` for formal
outputs rather than replacing unrelated factor columns.

### Code Style / 代码风格

中文：Python 沿用小模块、`snake_case`、现有 dataclass 配置与 `AlphaModel/FeatureProvider`
接口；兼容 Python 3.8.3，需要时使用 `from __future__ import annotations`，避免依赖新版
Python 才有的运行时 API。不要在 import 时启动 GPU、读全库或写文件。数值数组优先使用
NumPy/Arrow，跨日期缓存必须有明确容量和释放点；错误应包含日期、路径或配置项。
Rust 沿用 `Result`、request/report、公共 loader/writer 与 `cargo fmt` 风格；
新接口只扩展任务需要的范围，不混入无关架构重写。修改公共接口时同步测试、调用方和文档。

English: use small modules, snake_case, existing dataclass configs and model/provider
contracts. Preserve Python 3.8.3 compatibility, including runtime APIs; deferred
annotations alone do not make newer APIs compatible. Avoid import-time GPU work,
database scans and writes. Prefer NumPy/Arrow numerical paths, bounded caches and
contextual errors. In Rust, follow Result-based errors, request/report APIs,
shared loaders/writers and cargo fmt. Keep scope narrow and update callers,
tests and documentation together when changing shared interfaces.

## 7. Validation And Handoff / 验证与交接

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
& $py -m pytest ml_alpha/tests -q
cargo fmt --manifest-path factor_engine\Cargo.toml -- --check
cargo test --manifest-path factor_engine\Cargo.toml logsig

# Optional isolated CLI integration test; build the executable first.
cargo build --release --manifest-path factor_engine\Cargo.toml
$env:YQ_TEST_DERIVE_LOGSIG_EXE = (Resolve-Path factor_engine/target/release/yq-factor-engine.exe).Path
& $py -m pytest ml_alpha/tests/test_derived_logsig.py -q
Remove-Item Env:YQ_TEST_DERIVE_LOGSIG_EXE
```

中文交接清单：

1. 先看 Git 状态、现有 TOML、provider、模型及公共 runtime，不覆盖其他 agent 的改动。
2. 新公式/字段/单位不确定时确认；新增逻辑应放在对应层，避免在 CLI 或配置解析器塞计算。
3. 覆盖公式、日期/股票对齐、缺失值、边界、标签泄漏、保存恢复和分批等价性测试。
4. 只做少量交易日的隔离验证，不擅自全量生产、训练或清理正式数据。
5. 性能对比使用相同日期、股票、线程，区分冷启动/预热和计算/IO，先验证数值一致。
6. 清理本轮临时配置、日志、样本和模型；不删除源数据、正式模型或他人的产物。
7. 更新 README、registry 和相关 metadata 流程说明；需要生产 metadata 时使用明确命令。
8. 汇报改动、测试、限制与可运行命令；仅在用户要求时提交推送，确认工作区状态。

English handoff checklist:

1. Inspect Git, config, provider, model and runtime first; preserve others' work.
2. Confirm ambiguous formulas/units and keep responsibilities in their owning modules.
3. Test formulas, alignment, missing data, boundaries, leakage, persistence and batch invariance.
4. Use small isolated validation; do not launch full production/training without authorization.
5. Compare equivalent workloads and correctness before reporting speed or peak memory gains.
6. Clean only task-owned temporary products; preserve source/formal data and model configs.
7. Update docs/catalogs and document any metadata refresh needed.
8. Report verification and commands; commit/push when requested, checking the final status.

## 8. Source Map / 源码导航

- [Main README / 功能入口](../README.md)
- [Config parsing / 配置解析](../yq_ml_alpha/config.py)
- [CLI / 命令入口](../yq_ml_alpha/cli.py)
- [Dataset / 特征与标签对齐](../yq_ml_alpha/data/dataset.py)
- [Feature providers / 特征接口](../yq_ml_alpha/features/base.py)
- [Training windows / 训练调度](../yq_ml_alpha/pipelines/runtime.py)
- [Factor pipeline / 正式因子流程](../yq_ml_alpha/pipelines/factor.py)
- [Artifact manifests / 模型清单](../yq_ml_alpha/output/artifacts.py)
- [Daily writer / 日频合并写入](../yq_ml_alpha/output/daily_wide_writer.py)
- [Metadata / 因子元数据](../yq_ml_alpha/output/factor_metadata.py)
- [Logsignature config / 当前配置](../factors/logsig_alpha_v.toml)
- [Rust derive-logsig / 派生实现](../../factor_engine/src/derive/logsig.rs)
- [Rust derived storage / 派生写入](../../factor_engine/src/derive/storage.rs)
- [Rust Python extension / 中性化桥接](../../factor_engine/python/yq_factor_engine_py/src/lib.rs)
- [Factor development / Rust因子开发](../../factor_engine/docs/FACTOR_DEVELOPMENT_README.md)

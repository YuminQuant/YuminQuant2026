# yq_ml_alpha / Python ML Alpha

`ml_alpha` 鏄?YuminQuant 鐨?Python 鏈哄櫒瀛︿範 alpha 鍖呫€傚畠璇诲彇姝ｅ紡鍥犲瓙搴撴垨澶栭儴 alpha root锛屾寜閰嶇疆鍒囧垎璁粌/楠岃瘉/棰勬祴绐楀彛锛岃缁冩ā鍨嬶紝骞舵妸棰勬祴鍒嗘暟鍐欐垚鏍囧噯鏃ラ alpha parquet銆?

`ml_alpha` is the Python ML alpha package. It reads formal factors or external alpha roots, builds train/valid/predict windows from TOML configs, trains models, and writes predictions as standard daily alpha parquet files.

## End-To-End Factor Layer

Official end-to-end factors now live under `ml_alpha/factors/` and use factor ids as their primary names:

```text
ml_alpha/factors/e2e_fct_000001.toml
ml_alpha/factors/e2e_fct_000002.toml
ml_alpha/factors/e2e_fct_000003.toml
```

The `models/mdl_*.toml` files are model and experiment configs. They are not formal factor identities. A formal factor config uses `factor_id = "e2e_fct_000001"` and writes the same value as the output column into:

```text
data/factors/stock/daily/{year}/{trade_date}.parquet
data/factors/factor_metadata.parquet
```

Run a formal factor from the `ml_alpha` directory:

```powershell
python -m yq_ml_alpha run --config factors\e2e_fct_000002.toml
python -m yq_ml_alpha factor-run --config factors\e2e_fct_000002.toml
```

Model artifacts for formal factors are stored under `data/model_workspace/{factor_id}/...`.

## 蹇€熷紑濮?/ Quick Start

涓嶅畨瑁呭寘鏃讹紝浠?`ml_alpha` 鐩綍杩愯锛?

Run from the `ml_alpha` directory when you do not want to install the package:

```powershell
cd ml_alpha
python -m yq_ml_alpha run --config models\mdl_000001.toml
cd D:\yuminwu_workspace\Internship\YuminQuant
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 15 --start-date 20101201 --end-date 20260424

`derive-bar` accepts stock minute bar sizes that divide 240 and satisfy `1 < bar_size <= 120`; `120` means one morning bar and one afternoon bar.

cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha
python -m yq_ml_alpha factor-run --config factors\e2e_fct_000001.toml
python -m yq_ml_alpha model-run --config models\experiments\monthly_xgb_36.toml
python -m yq_ml_alpha model-run --config models\experiments\monthly_mlp_36.toml
python -m yq_ml_alpha model-run --config models\experiments\monthly_elstm_ranknet_36.toml
```

鏈満 Python 3.8.3 GPU 鐜锛?
Python 3.8.3 GPU environment on this machine:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000001.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000002.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000003.toml
```

`D:\Users\Devin\anaconda383\python.exe` is Python 3.8.3 with PyTorch, CUDA,
pandas, and pyarrow available. Running from `ml_alpha` avoids changing
`PYTHONPATH` or global environment variables.


濡傛灉鎯充粠浠撳簱鏍圭洰褰曠洿鎺ヨ繍琛岋紝鍙互涓存椂璁剧疆 `PYTHONPATH`锛?

If running from the repository root, set `PYTHONPATH` temporarily:

```powershell
$env:PYTHONPATH = "D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha"
python -m yq_ml_alpha run --config ml_alpha\models\experiments\monthly_mlp_36.toml
```

鍙敤瀛愬懡浠?/ Commands:

```powershell
python -m yq_ml_alpha run --config models\experiments\monthly_mlp_36.toml
python -m yq_ml_alpha train --config models\experiments\monthly_mlp_36.toml
python -m yq_ml_alpha predict --config models\experiments\monthly_mlp_36.toml
python -m yq_ml_alpha materialize --config models\experiments\monthly_mlp_36.toml
python -m yq_ml_alpha model-run --config models\experiments\monthly_mlp_36.toml
```

`run` = train + predict + write alpha. `train` only saves model artifacts. `predict` uses existing artifacts. `materialize` only builds sample cache when configured.

## 杈撳嚭涓庡洖娴?/ Output And Backtest

Alpha 杈撳嚭 / Alpha output:

```text
data/models/{year}/{trade_date}.parquet
columns: trade_date, ts_code, alpha_id
```

鍚屼竴澶╁涓?alpha 浼氬啓鍏ュ悓涓€涓?daily parquet銆侾arquet 涓嶈兘鐪熸鍘熷湴杩藉姞鍒楋紝鍥犳 writer 浼氳鍙栨棫鏂囦欢銆佸悎骞?瑕嗙洊褰撳墠 alpha 鍒楋紝鍐嶉噸鍐欐枃浠躲€?

Multiple alphas for the same date are stored in one daily parquet. Parquet cannot append a column in place, so the writer reads, merges, and rewrites the file.

鍥炴祴 / Backtest:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\factors --groups 10 --rebalance 20
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_monthly_alpha --factor-root data\factors --factor-fill ffill
```

浣庨 alpha 鍙湪鏈堟湯鎴栧懆鏈湁鎴潰鏃讹紝浣跨敤 `--factor-fill ffill` 璁╁洖娴嬬敤鏈€杩戜竴娆?alpha 鏃ラ缁撶畻銆?

Use `--factor-fill ffill` for low-frequency alpha snapshots.

## 閰嶇疆缁撴瀯 / TOML Config

鎵€鏈夌ず渚嬮厤缃兘鍦細

Configs live under:

```text
ml_alpha/models/*.toml
ml_alpha/models/experiments/*.toml
ml_alpha/factors/*.toml
```

Numbered production model configs use `mdl_******` ids. The registry file
`ml_alpha/model_registry.toml` records the model id, output alpha id, config
path, model class, description, feature source, preprocessing, and tags.

鍏稿瀷 factor-frame 閰嶇疆 / Typical factor-frame config:

```toml
run_id = "mdl_000001"
alpha_id = "mdl_000001"
data_root = "data"
output_root = "data/models"

[dates]
train = [20110101, 20260424]
valid = []
predict = [20110101, 20260424]

[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "rolling"               # static | expanding | rolling
refit_frequency = "monthly_end"
train_sample_count = 36         # fixed sample-count mode
validation_sample_count = 0

[label]
id = "future_vwap_return_20d"

[filters]
exclude_limit = false
exclude_st = true
exclude_bj = true

[preprocess]
cross_section_transform = "rank_gauss"
feature_fill_value = 0.0

[features]
type = "factor_frame"          # factor_frame | bar_panel | multi_bar_panel
root = "data/factors/stock/daily"
columns = "__all__"

[model]
name = "linear"
class = "yq_ml_alpha.models.linear_model.LinearRegressionAlphaModel"
artifact_dir = "data/model_workspace/mdl_000001/artifacts"
```

甯哥敤閲囨牱棰戠巼 / Sampling frequencies:

```text
daily
weekly
monthly_end
quarterly
5
20
every_5_days
```

`monthly_end` 鍙栨瘡涓嚜鐒舵湀鏈€鍚庝竴涓氦鏄撴棩銆俙"20"` 鍜?`every_20_days`
浼氬厛鍙栬缁冨尯闂村唴鐨勪氦鏄撴棩鍒楄〃锛屽啀鎵ц `dates[::20]`銆?
`monthly_end` selects the last trading day of each calendar month. `"20"` and
`every_20_days` build the trading-day list first, then apply `dates[::20]`.

### 璁粌绐楀彛璇箟 / Training Window Semantics

`rolling` 鍙〃绀衡€滄寜 `refit_frequency` 鍛ㄦ湡閲嶆柊璁粌鈥濄€傜湡姝ｈ鍙栧摢浜涜缁冩牱鏈紝鐢变笅闈㈠嚑绉嶄簰鏂ラ厤缃ā寮忓喅瀹氾細

`rolling` only means "refit on the `refit_frequency` schedule". The training
samples are selected by one of the mutually exclusive configuration modes below.

鍥哄畾鎴潰鏁版ā寮忛€傚悎鏈堟湯鎴潰妯″瀷锛屼緥濡傜嚎鎬фā鍨嬶細

Fixed sample-count mode is used by month-end snapshot models, such as linear
models:

```toml
[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "monthly_end"
train_sample_count = 36
validation_sample_count = 1
```

姣忎釜 refit date 鍓嶏紝鍏堟寜 `train_frequency` 寰楀埌閲囨牱鎴潰锛岀劧鍚庡彇鏈€杩?`36 + 1` 涓埅闈細鍓?36 涓缁冿紝鏈€鍚?1 涓獙璇併€傝繖閲屼笉浣跨敤
`train_lookback`銆?
Before each refit date, the pipeline samples dates by `train_frequency`, takes
the latest `36 + 1` snapshots, uses the first 36 for training and the last one
for validation. This mode does not use `train_lookback`.

鏃ユ湡鍥炵湅 + 姣斾緥楠岃瘉妯″紡閫傚悎绔埌绔?GRU锛?
Lookback + validation-ratio mode is used by end-to-end GRU models:

```toml
[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "semiannual_end"
train_lookback = "3y"
validation_ratio = 0.2
```

姣忎釜 refit date 鍓嶏紝鍏堢敓鎴愯缁冩棩鏈熺獥鍙ｏ紝鍐嶅湪绐楀彛鍐呮寜 `train_frequency`
閲囨牱锛屾渶鍚庢寜鏃堕棿椤哄簭鍒囧垎 train/valid銆俙validation_ratio=0.2` 鐨勪緥瀛愶細

For each refit date, the pipeline first builds a training date window, samples
inside that window, then splits train/valid chronologically. Examples with
`validation_ratio=0.2`:

```text
36 sampled dates -> 29 train / 7 valid
35 sampled dates -> 28 train / 7 valid
34 sampled dates -> 27 train / 7 valid
2 sampled dates  -> 1 train / 1 valid
```

`train_lookback` 鏀寔鏁存暟骞存垨鏁存暟浜ゆ槗鏃ワ細

`train_lookback` supports integer years or integer trading days:

```toml
train_lookback = "3y"    # natural-year lookback
train_lookback = "756d"  # trading-day lookback
```

`3y` 鏄嚜鐒跺勾鍥炵湅銆備緥濡?refit anchor 涓?`20160630` 鏃讹紝璁粌缁撴潫鏃ユ槸
refit 鍓嶄竴涓氦鏄撴棩 `20160629`锛屼笁骞村洖鐪嬬獥鍙ｇ害涓?`20130701..20160629`銆?`756d` 鏄氦鏄撴棩鍥炵湅锛屼粠璁粌缁撴潫鏃ュ線鍓嶆暟 756 涓氦鏄撴棩銆傚綋鍓嶅疄鐜板彧鎺ュ彈
鏁存暟锛屼笉鎺ュ彈 `0.5y`锛涢渶瑕佸崐骞存椂寤鸿鍐欐垚浜ゆ槗鏃ヨ繎浼硷紝渚嬪 `120d`銆?
`3y` is a calendar-year lookback. For refit anchor `20160630`, the train end is
the previous trading day `20160629`, and the three-year window is approximately
`20130701..20160629`. `756d` counts 756 trading days backward from the train
end. Fractional values such as `0.5y` are not supported; use a trading-day
approximation such as `120d` instead.

閰嶇疆浜掓枼瑙勫垯 / Conflict rules:

```text
train_lookback cannot be used with train_sample_count
validation_ratio cannot be used with validation_sample_count
validation_ratio cannot be used with train_sample_count
static cannot use train_lookback or validation_ratio
rolling + validation_ratio requires train_lookback
```

`static` 浣跨敤 `[dates].train` 鍜?`[dates].valid`锛屼笉浼氭寜 refit 婊氬姩锛沗expanding`
涓嶉厤缃?`train_lookback` 鏃朵粠 `[dates].train[0]` 鎵╁睍鍒?refit 鍓嶄竴鏃ワ紝閰嶇疆
`train_lookback` 鏃跺垯鍙娇鐢ㄥ洖鐪嬬獥鍙ｃ€?
`static` uses `[dates].train` and `[dates].valid` directly and does not refit
over time. `expanding` without `train_lookback` expands from `[dates].train[0]`
to the day before each refit; with `train_lookback`, it uses the lookback window.

`valid = []` 琛ㄧず涓嶄娇鐢ㄥ浐瀹氶獙璇佸尯闂淬€傚姩鎬侀獙璇侀泦鐢?`validation_sample_count` 鎴?`validation_ratio` 鍐冲畾銆?
`valid = []` means no fixed validation period. Dynamic validation is controlled
by `validation_sample_count` or `validation_ratio`.

### 婊氬姩璁粌缁窇 / Resuming Rolling Training

濡傛灉涓€娆?rolling 璁粌涓€旀殏鍋滐紝閫氬父**涓嶈**涓轰簡缁窇鑰屾妸 `train` 鏀规垚鏆傚仠鏃ユ湡闄勮繎銆俙train` 鏄缁冩牱鏈睜涓婇檺锛岀▼搴忕湡姝ｈ鍙栧摢浜涜缁冩埅闈㈢敱姣忎釜 refit window 鐨?`train_dates` 鍐冲畾銆傚浐瀹氭埅闈㈡暟妯″紡浼氬彇 refit 涔嬪墠鏈€杩?N 涓噰鏍锋埅闈紱`train_lookback + validation_ratio` 妯″紡浼氬厛鍥炵湅绐楀彛锛屽啀閲囨牱鍜屽垏鍒嗐€?
缁窇鏃跺簲涓昏璋冩暣 `predict` 鍖洪棿锛岃瀹冧粠鈥滀笅涓€娈甸娴嬫墍闇€鐨?refit anchor鈥濆紑濮嬨€備緥濡傚凡缁忛娴嬪畬 `20231229`锛屼笅涓€娈甸渶瑕侀娴?2024 骞?1 鏈堬紝鏈堥 refit 涓嬪缓璁細

```toml
[dates]
train = [20110101, 20260424]   # 淇濈暀瓒冲鍘嗗彶鏍锋湰姹?
valid = []
predict = [20231229, 20260424] # 20231229 鏄笅涓€娈甸娴嬬殑 refit anchor
```

褰撳墠瀹炵幇涓紝`refit_date` 鏈韩涓嶄細琚绐楀彛閲嶆柊棰勬祴锛涚獥鍙ｉ娴嬬殑鏄?`refit_date` 涔嬪悗鍒颁笅涓€涓?refit date 涔嬮棿鐨勪氦鏄撴棩銆傚洜姝や笂闈㈢殑閰嶇疆浼氫粠 `20240102` 寮€濮嬪啓鍚庣画 alpha锛屽悓鏃朵繚鐣欒冻澶熷巻鍙叉牱鏈敤浜?rolling 璁粌銆?

When resuming an interrupted rolling run, usually do **not** shrink `train` to the interrupted date. `train` is the upper bound of the sample pool. The actual training data is selected per refit window from `window.train_dates`. Fixed sample-count mode uses the latest N sampled snapshots before the refit date; `train_lookback + validation_ratio` mode first builds a lookback window, then samples and splits it.

To resume, adjust `predict` to start from the refit anchor that owns the next unfinished prediction segment. If predictions are complete through `20231229` and the next segment is January 2024, keep `train` broad and set `predict = [20231229, 20260424]`. The refit date itself is not predicted again; predictions start after it.

## 棰勫鐞?/ Preprocessing

褰撳墠鎺ㄨ崘鎴潰鍙樻崲鏄細

Recommended cross-sectional transform:

```toml
[preprocess]
cross_section_transform = "rank_gauss"
feature_fill_value = 0.0
```

`rank_gauss` 鍋氾細

`rank_gauss` applies:

```text
rank -> (rank - 0.5) / n -> inverse normal CDF -> cross-section zscore
```

feature 缂哄け鍊煎湪鍙樻崲鍚庡～ `feature_fill_value`锛宭abel 缂哄け涓嶄細濉厖锛岃缁冩椂缂?label 鐨勬牱鏈細琚墧闄ゃ€?

Feature missing values are filled after transform. Label missing values are not filled and are dropped for training.

鍙敤 transform 鍦?`yq_ml_alpha/features/transforms.py` 娉ㄥ唽銆傛柊澧?transform 鏃跺彧闇€瑕佹敞鍐屼竴娆★紝鐒跺悗 TOML 涓啓娉ㄥ唽鍚嶃€?

Transforms are registered in `yq_ml_alpha/features/transforms.py`. Add a transform once and reference its registered name in TOML.

## 宸叉湁妯″瀷 / Built-In Models

```text
LinearRegressionAlphaModel                  mdl_000001.toml
XGBoostAlphaModel                           monthly_xgb_36.toml
XGBoostOptunaAlphaModel                     monthly_xgb_optuna_36.toml
LightGBMOptunaAlphaModel                    monthly_lgbm_optuna_36.toml
LassoAlphaModel                             mdl_000002.toml
RidgeAlphaModel                             mdl_000003.toml
ElasticNetAlphaModel                        mdl_000004.toml
PCAOLSAlphaModel                            mdl_000005.toml
BarGRUAlphaModel                            e2e_fct_000001.toml
MultiBarGRUAlphaModel                       e2e_fct_000002.toml
ResidualMultiBarGRUAlphaModel               e2e_fct_000003.toml
RandomForestAlphaModel                      monthly_rf_36.toml
MLPAlphaModel                               monthly_mlp_36.toml
RNNAlphaModel                               monthly_rnn_36.toml
GRUAlphaModel                               monthly_gru_36.toml
CNNAlphaModel                               monthly_cnn_36.toml
eLSTMRankNetAlphaModel                      monthly_elstm_ranknet_36.toml
ICSignEqualWeightAlphaModel                 monthly_ic_sign_equal_weight.toml
MeanFeatureAlphaModel                       mean_combo_smoke.toml
```

娣卞害妯″瀷渚濊禆 PyTorch锛沊GBoost/LightGBM/Optuna 鏄彲閫変緷璧栥€?

Deep models require PyTorch. XGBoost, LightGBM, and Optuna are optional dependencies.

## Sequence 妯″瀷杈撳叆 / Sequence Model Input

`RNN/LSTM/GRU/eLSTM` 浣跨敤 `DatasetBuilder.load_sequence()` 璇诲彇杩囧幓 `sequence_length` 涓牱鏈埅闈€傝嫢 `sequence_length = 6`锛岃缁冩牱鏈棩鏈熸槸鏈堟湯锛屽垯姣忎釜鏍锋湰鍖呭惈鏈€杩?6 涓湀鏈埅闈㈢殑 feature銆?

`RNN/LSTM/GRU/eLSTM` use `DatasetBuilder.load_sequence()` to load the last `sequence_length` sample dates. With `sequence_length = 6` and monthly samples, each row contains features from the last six month-end snapshots.

杩涘叆妯″瀷鍓嶇殑褰㈢姸锛?

Input shape before model:

```text
flat DataFrame feature matrix: [N, sequence_length * F]
torch tensor:                  [N, sequence_length, F]
```

濡傛灉 feature 鏁颁笉鑳芥暣闄?`sequence_length`锛屼細鍦ㄥ彸渚цˉ 0 鍚?reshape銆?

If feature count is not divisible by `sequence_length`, zeros are padded on the right before reshaping.

## Bar Panel End-to-End GRU / 閫氱敤 Bar Panel 绔埌绔ā鍨?
### 璁捐 / Design

`bar_panel` 鏄鍒扮閲忎环妯″瀷鐨勯€氱敤琛屾儏杈撳叆鎺ュ彛銆傚畠璐熻矗鎶婂師濮嬭鎯?bar
鍚堟垚涓哄浐瀹氶暱搴︾殑 tensor 鐗瑰緛锛屼絾涓嶄細鎶婂悎鎴愬悗鐨?bar 浣滀负鐙珛鏂囦欢钀界洏銆?
`bar_panel` is the reusable market-bar input interface for end-to-end models. It
builds fixed-length tensor features from raw bars, but it does not persist the
synthesized bars as standalone files.

鍒嗛挓婧愭暟鎹祦绋?/ Minute source flow:

```text
璇诲彇鏌愪竴澶?1m parquet / read one daily 1m parquet
  -> 杩囨护 .BJ銆佸綋鏃?ST銆?9:30 琛?/ filter .BJ, same-day ST, and 09:30 rows
  -> groupby(ts_code).resample(...) 鍚堟垚鐩爣 bar / aggregate with pandas resample
  -> 鍙湪杩涚▼鍐?LRU cache 淇濈暀鍚堟垚鍚庣殑鏃ュ害 bar / keep aggregated day in cache
  -> 閲婃斁鍘熷 1m DataFrame / release raw 1m DataFrame
  -> 璇诲彇涓嬩竴澶?/ read the next day
```

鍒嗛挓 resample 鍙ｅ緞 / Minute resample rule:

```python
df = df.sort_values(["ts_code", "trade_time"])
bars = (
    df.set_index("trade_time")
      .groupby("ts_code")
      .resample(
          f"{bar_size}min",
          origin="start_day",
          offset="9h30min",
          label="right",
          closed="right",
      )
      .agg({
          "open": "first",
          "high": "max",
          "low": "min",
          "close": "last",
          "vol": "sum",
          "amount": "sum",
      })
      .dropna(subset=["open"])
      .reset_index()
)
```

`strict=true` 琛ㄧず姣忓彧鑲＄エ蹇呴』鏈夊畬鏁寸殑鏍囧噯 bar 搴忓垪锛涗笉鍐嶈姹傛瘡鏍?bar
鍐呴儴鍒氬ソ鏈?`bar_size` 鏉?1m 鏁版嵁銆俙bar_size=15` 鏃讹紝鏍囧噯 A 鑲℃棩鍐呮牱鏈?浼氬緱鍒?16 鏍?15 鍒嗛挓 bar銆?
With `strict=true`, each stock must have the full canonical bar sequence. The
implementation no longer requires each bar to contain exactly `bar_size` one-minute
rows. With `bar_size=15`, a standard A-share session yields 16 15-minute bars.

鏃ラ婧愭暟鎹祦绋?/ Daily source flow:

```text
鎸?trade_date 璇诲彇 daily pv parquet / read daily pv parquet by trade_date
  -> 濡傛灉 bar_size > 1锛屽悎鎴愰潪閲嶅彔 N 鏃?bar / aggregate non-overlapping N-day bars
  -> 鍦ㄨ繘绋嬪唴 cache 淇濈暀鍚堟垚鍚庣殑 daily panel / keep aggregated daily panel in cache
```

`cache_samples = true` 鍙細缂撳瓨鏈€缁堣缁?棰勬祴鏍锋湰锛屾柟渚?debug锛涘畠涓嶆槸鍚堟垚
bar 鐨勬寔涔呭寲缂撳瓨銆?
`cache_samples = true` only caches final train/predict samples for debugging. It
is not a persistent cache of synthesized source bars.

### Single Bar Panel: `e2e_fct_000001`

`e2e_fct_000001` 鏄崟棰戠巼 15 鍒嗛挓 GRU 妯″瀷銆傛牳蹇冮厤缃涓嬶細

`e2e_fct_000001` is the single-frequency 15-minute GRU end-to-end factor. Core config:

```toml
[features]
type = "bar_panel"
root = "data/derived/stock/bar/15m"
columns = ["open", "high", "low", "close", "vwap", "volume"]

[features.params]
source_frequency = "minute_bar"
bar_size = 15
lookback_sessions = 20
time_series_scale = "mean"
strict = true
```

妯″瀷杈撳叆 / Model input:

```text
[N, 320, 6] = [stocks, 20 trading days * 16 bars, open/high/low/close/vwap/volume]
```

棰勫鐞嗛『搴?/ Preprocessing order:

1. 瀵规瘡鍙偂绁ㄣ€佹瘡涓壒寰侊紝鎶?320 姝ユ椂搴忓€奸櫎浠ヨ嚜韬椂搴忓潎鍊硷紱
2. 瀵规瘡涓?`trade_date`锛屽皢姣忎釜 `time_step x feature` 鍒楀仛鎴潰 z-score锛?3. 瀵?`future_vwap_return_20d` label 鍋氭埅闈?z-score銆?
English: divide each stock-feature time series by its own mean, then apply
cross-sectional z-score to each `time_step x feature` column and to the label.

璁粌绐楀彛 / Training window:

```toml
[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "semiannual_end"
train_lookback = "3y"
validation_ratio = 0.2
```

鍚箟鏄細姣忓崐骞撮噸鏂拌缁冧竴娆★紱姣忔 refit 鍓嶅洖鐪嬩笁骞翠氦鏄撳巻鍙诧紱鍦ㄤ笁骞寸獥鍙ｅ唴姣?20
涓氦鏄撴棩閲囨牱涓€涓缁冩埅闈紱鍐嶆寜鏃堕棿椤哄簭鍋?80/20 train/valid 鍒囧垎锛況efit
涔嬪悗鍒颁笅涓€娆?refit 鍓嶆瘡鏃ラ娴嬨€?
This means: refit every half-year; look back three years before each refit;
sample one training snapshot every 20 trading days inside that window; split
the sampled dates chronologically into 80/20 train/valid; predict daily until
the next refit.

### Multi Bar Panel: `e2e_fct_000002` / `e2e_fct_000003`

`multi_bar_panel` 鐢ㄦ潵缁勫悎澶氫釜 `bar_panel`銆傚綋鍓嶇敓浜ч厤缃娇鐢ㄤ竴涓棩棰戝垎鏀?鍜屼竴涓?15 鍒嗛挓鍒嗘敮锛?
`multi_bar_panel` composes multiple `bar_panel` providers. Current production
configs use one daily branch and one 15-minute branch:

```toml
[features]
type = "multi_bar_panel"

[features.panels.daily]
root = "data/stock_data/daily/pv"
source_frequency = "daily"
bar_size = 1
lookback_sessions = 40
time_series_scale = "last"
columns = ["open", "high", "low", "close", "vwap", "volume"]

[features.panels.minute]
root = "data/derived/stock/bar/15m"
source_frequency = "minute_bar"
bar_size = 15
lookback_sessions = 20
time_series_scale = "mean"
columns = ["open", "high", "low", "close", "vwap", "volume"]
```

杈撳嚭鍒椾細甯﹀垎鏀墠缂€锛屼緥濡?`daily__open__t000` 鍜?`minute__close__t319`銆?妯″瀷渚ф寜鍓嶇紑鎭㈠ tensor锛?
English: output columns are prefixed by branch, such as `daily__open__t000`
and `minute__close__t319`; the model restores tensors by prefix.

```text
daily branch:  [N, 40, 6]
minute branch: [N, 320, 6]
```

`e2e_fct_000002` 鏄櫘閫氬棰戠巼娣峰悎 GRU锛氭棩棰戝垎鏀拰鍒嗛挓鍒嗘敮鍒嗗埆杈撳嚭 30 缁磋〃绀猴紝
缁忚繃 BatchNorm 鍚?concat锛屽啀閫氳繃 FC 鏄犲皠涓轰竴涓?score銆?
`e2e_fct_000002` is the normal multi-frequency GRU. The daily and minute branches
produce 30-dimensional representations, then BatchNorm + concat + FC maps them
to one score.

`e2e_fct_000003` 鏄弬鏁板喕缁?+ 娈嬪樊棰勬祴鐗堟湰锛?
`e2e_fct_000003` is the frozen-parameter residual version:

```text
stage 1: train daily branch only -> y_hat_1
stage 2: freeze daily branch, train minute branch -> y_hat_2
final:   y_hat = y_hat_1 + y_hat_2
```

涓や釜闃舵閮戒娇鐢?date-wise negative IC loss銆俙loss_history.parquet` 浼氬寘鍚?`stage = stage1_daily / stage2_residual`銆?
Both stages use date-wise negative IC loss. `loss_history.parquet` includes
`stage = stage1_daily / stage2_residual`.

`e2e_fct_000002` 鍜?`e2e_fct_000003` 浣跨敤涓?`e2e_fct_000001` 鐩稿悓鐨勫崐骞村害 refit銆?`train_lookback="3y"`銆乣train_frequency="20"`銆乣validation_ratio=0.2`
璁粌绐楀彛璇箟銆?
`e2e_fct_000002` and `e2e_fct_000003` use the same semiannual refit,
`train_lookback="3y"`, `train_frequency="20"`, and `validation_ratio=0.2`
window semantics as `e2e_fct_000001`.

### 璁粌鍛戒护 / Training Commands

榛樿 Python 鐜 / Default Python environment:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

python -m yq_ml_alpha factor-run --config factors\e2e_fct_000001.toml
python -m yq_ml_alpha factor-run --config factors\e2e_fct_000002.toml
python -m yq_ml_alpha factor-run --config factors\e2e_fct_000003.toml
```

Python 3.8.3 GPU 鐜 / Python 3.8.3 GPU environment:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000001.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000002.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha factor-run --config factors\e2e_fct_000003.toml
```

### 鍥炴祴鍛戒护 / Backtest Commands

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant

cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors e2e_fct_000001 --factor-root data\factors --groups 10 --rebalance 20 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors e2e_fct_000002 --factor-root data\factors --groups 10 --rebalance 20 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors e2e_fct_000003 --factor-root data\factors --groups 10 --rebalance 20 --date-batch-size 120
```

## Diagnostics / Loss 杈撳嚭

MLP銆丷NN銆丩STM銆丟RU銆乪LSTM RankNet 鏀寔 diagnostics銆俆OML 涓墦寮€锛?

MLP, RNN, LSTM, GRU, eLSTM RankNet, and BarGRU support diagnostics:

```toml
[diagnostics]
enabled = true
print_epoch = true
write_loss_history = true
write_model_info = true
write_window_summary = true
```

绐楀彛绾ц緭鍑?/ Per-window outputs:

```text
data/model_workspace/{run_id}/artifacts/{window_id}/loss_history.parquet
data/model_workspace/{run_id}/artifacts/{window_id}/model_info.json
```

鍏?run 姹囨€?/ Run-level summaries:

```text
data/model_workspace/{run_id}/diagnostics/loss_history.parquet
data/model_workspace/{run_id}/diagnostics/window_summary.parquet
```

Regularized linear models also write diagnostics when enabled. For
`mdl_000002` / `mdl_000003` / `mdl_000004`, the aggregate
`window_summary.parquet` includes `best_alpha`, `best_l1_ratio` when
applicable, `best_score`, and `best_params_json`.

`mdl_000005` writes PCA diagnostics to the same summary file, including
`n_original_features`, `n_components`, `explained_variance_ratio_sum`, and
`explained_variance_ratio_json`.

`e2e_fct_000001` writes GRU diagnostics to the same paths. `loss_history.parquet`
contains date-wise negative IC loss by epoch, and `model_info.json` records the
15-minute panel shape, train/valid row counts, device, and best epoch.

`loss_history.parquet` 璁板綍姣忎釜 epoch 鐨?`train_loss`銆乣valid_loss`銆乣best_loss`銆乣stale_epochs`銆乣elapsed_seconds` 绛夈€俙model_info.json` 璁板綍鏍锋湰閲忋€佽澶囥€佹ā鍨嬪弬鏁般€乥est epoch 鍜?best loss銆?

`loss_history.parquet` records per-epoch loss. `model_info.json` records data sizes, device, model params, best epoch, and best loss.

## 璋冨弬 / Tuning

璋冨弬灞炰簬妯″瀷鍐呴儴閫昏緫锛屼笉鍋氭鏋剁骇鍏叡 objective/loss銆俆OML 涓€氳繃 `[model.search]` 鍜?`[model.search.space]` 鏆撮湶鎼滅储绌洪棿銆?

Tuning is model-owned. There is no shared framework-level objective or loss. Search spaces are configured through `[model.search]` and `[model.search.space]`.

绀轰緥 / Example:

```toml
[model.search]
enabled = true
method = "random"
n_iter = 40
scoring = "neg_mean_squared_error"

[model.search.space]
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
solver_selection = ["cyclic", "random"]
```

鍏抽棴璋冨弬鏃讹細

Disable tuning:

```toml
[model.search]
enabled = false
```

## 鏂板妯″瀷 / Add A Model

鏂板鏂囦欢锛?

Create a model file:

```text
ml_alpha/yq_ml_alpha/models/my_model.py
```

瀹炵幇鎺ュ彛 / Implement:

```python
from yq_ml_alpha.models.base import AlphaModel, ModelContext

class MyAlphaModel(AlphaModel):
    def fit(self, train_data, valid_data, context: ModelContext) -> None:
        ...

    def predict(self, data, context: ModelContext):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

TOML 涓寚鍚戠被璺緞锛?

Reference the class path in TOML:

```toml
[model]
name = "my_model"
class = "yq_ml_alpha.models.my_model.MyAlphaModel"
artifact_dir = "data/model_workspace/my_run/artifacts"

[model.params]
learning_rate = 0.03
```

璁粌绠＄嚎浼氬姩鎬?import 璇ョ被銆傛ā鍨嬭嚜宸辩殑 loss銆佽皟鍙傘€乪arly stopping 鍜?artifact 缁撴瀯鐢辨ā鍨嬪唴閮ㄥ喅瀹氥€?

The training pipeline dynamically imports the class. Loss, tuning, early stopping, and artifacts are model-owned.

## IC Sign Equal Weight 妯″瀷

`ICSignEqualWeightAlphaModel` 璇诲彇 Rust 鍥炴祴杈撳嚭鐨?RankIC 搴忓垪锛屼娇鐢?`sign(mean(rank_ic))` 璋冩暣姣忎釜 feature 鏂瑰悜锛屽啀瀵规湁鏁?feature 绛夋潈骞冲潎锛?

`ICSignEqualWeightAlphaModel` reads RankIC history from Rust backtest outputs, orients each feature by `sign(mean(rank_ic))`, then averages valid features:

```toml
[model]
class = "yq_ml_alpha.models.ic_sign_model.ICSignEqualWeightAlphaModel"

[model.params]
ic_root = "data/backtest/stock/daily/ic"
ic_metric = "rank_ic"
```

缂?IC 鏂囦欢銆丷ankIC 鍏ㄧ┖鎴栧潎鍊间负 0 鐨?feature 浼氳鍓旈櫎銆?

Features with missing or invalid IC files are dropped.

## 缁存姢鎻愮ず / Maintenance Notes

- 褰撳墠閰嶇疆閮藉湪 `ml_alpha/models/*.toml and ml_alpha/models/experiments/*.toml`銆?
- `data/models` 鏄寮?ML alpha 杈撳嚭鏍圭洰褰曘€?
- `data/model_workspace/{run_id}` 鏄ā鍨?artifact銆乨iagnostics 鍜屽彲閫?cache 鐩綍銆?
- `rank_gauss` 鏄綋鍓嶆帹鑽愰澶勭悊 transform銆?
- 杈撳嚭 alpha 涓嶅啓鍏?Rust factor metadata锛屽洖娴嬫椂鐢?`--factor-root data\factors --factors alpha_id`銆?

- Current model configs live under `ml_alpha/models/*.toml` and `ml_alpha/models/experiments/*.toml`.
- `data/models` is the formal ML alpha output root.
- `data/model_workspace/{run_id}` stores artifacts, diagnostics, and optional cache.
- `rank_gauss` is the recommended transform.
- ML alpha is not written into Rust factor metadata; use `--factor-root data\factors`.


# DTAN：双轨对抗因子挖掘

参考西南证券2026-06-15《双轨最优传输分布纠偏与对抗生成式因子挖掘模型》。正式因子为 `dtan_alpha`，配置 `factors/dtan_alpha.toml`，券商tag为 `XNZQ`。

直接实现最终的Dual-Track Adversarial Network。输出分布和潜空间两条轨道均使用WGAN-GP Critic，不再叠加前代GDA/UOA/DTSN的手工UOT或切片Wasserstein损失。

## 输入与项目口径

复用LSG的 `market_style_sequence` 公共输入协议：OHLC、VWAP、volume、turnover_rate_f，加本地9项CNE6一级因子，共16维。Barra字段为DIVIDEND_YIELD、GROWTH、LIQUIDITY、MOMENTUM、QUALITY、SENTIMENT、SIZE、VALUE、VOLATILITY；不接入EMA、DIF、DEA、MACD。

本模型回看40个交易日。价格复权后除以样本末日复权收盘价，成交量和换手除以各自40日均值，再在每个历史位置按完整样本日股票集合做截面z-score；Barra仅做截面z-score，缺失填0。股票池、ST/北交所过滤、零成交量VWAP=0、完整行情历史规则与 [LSG](LSG.md) 的同类处理一致。

行情、基础数据、复权因子和CNE6直接读取已有日表，无需新增派生数据。每日连接结果使用 1 GiB 的公共 RAM 缓存。HDFS 读取采用部署并发配置，本地串行。窗口依赖的归一化在编码前按完整截面计算，输入归一化不读取标签。禁止磁盘样本缓存和 `factor-materialize`。

- 年度重训，近60个月按57个月训练/3个月验证划分；固定5日采样，每轮遍历全部训练日期。
- 2011起训练、2016起逐年预测至2026-04-24；10日VWAP标签，公共层按11个交易日到期边界清洗。
- batch_size=5000仅限制生成器股票编码块；IC和输出轨按完整有效标签截面计算。
- 单种子42，最多100轮，验证IC连续10轮未改善早停。最终保存最优验证IC的生成器。

## 网络和损失

生成器接收 `[stocks, 40, 16]` 的实际特征序列，不接收随机先验向量。线性投影到64维、加入正弦位置编码，经两个四头Transformer编码层（FFN128、Dropout0.1）：

1. 取末步64维表示，经64→32→1的ReLU预测头输出收益分数。
2. 展平全部40×64维表示，线性映射为32维潜特征，约束整段历史的表征。

输出Critic DO为1→128→128→1；潜空间Critic DL为32→128→128→1。首个隐藏层使用LayerNorm，激活为LeakyReLU(0.2)，末层无Sigmoid。

研报理论章节要求避免BatchNorm以保持逐样本梯度惩罚有效，实证章节却采用BatchNorm。按用户确认采用LayerNorm。图21中生成器损失第二个Critic名称重复为DO，按双轨定义及正文Step5使用DL。

真实收益仅对已观测标签做MAD5缩尾和截面z-score。每个训练日依次执行：

- 生成器首次前向在no_grad下取得假收益、假潜特征，更新Critic时显式detach假样本。
- DO更新一次，目标 `mean(DO(fake)) - mean(DO(real)) + 10 * GP`。
- Track2激活后DL更新一次，采用相同WGAN-GP形式。对真假样本按行随机线性插值，惩罚逐样本输入梯度范数偏离1的平方均值。
- 冻结两个Critic的参数，再次前向生成器，最小化 `-0.1*mean(DO(fake)) - 0.1*mean(DL(fake)) + 1*(1-IC)`。预热期跳过DL项。

两个轨道的真假样本分别加入标准差0.01的独立高斯噪声，生成器对抗步骤的假样本同样加噪；IC始终使用干净预测。G与Critic均采用Adam、lr=1e-4，G betas=(0.9,0.999)，Critic betas=(0,0.9)，梯度范数裁剪5。

噪声标准差、损失权重、GP权重、Transformer细节和优化器细节是用户确认的工程设定，均在TOML中显式配置，不宣称是研报完整披露参数。预测时关闭Dropout，不加噪声。

## 训练专用潜特征库

前5个epoch库为空，DL不更新，只有DO与IC参与。第6轮开始前首次建库，此后第11、16……轮开始前重填：

1. 只从训练集各日期的有效真实收益标签中选Top10%。验证和预测数据不进入候选接口。
2. 从全部“股票×训练日期”候选中均匀无放回抽取最多2048条，不按日期抽取训练子集。
3. 用当时生成器的eval模式重新编码被选样本，生成32维潜特征；不继承旧版本编码。
4. 用新的脱离计算图的CPU float32快照替换旧库，冻结至下次重填。记录版本、候选数、条目数和来源日期范围。

该处理采用DTAN的周期重填规则，不沿用前代DTSN“每3个batch将预测Top10%追加FIFO”的更新规则。真实标签选股和跨候选均匀抽样属于用户确认的接入口径。

容量2048小于部分日期的股票数时，DL从当前股票潜特征和库中分别无放回抽取 `min(当前股票数, 库条目数)` 条；这只限制潜空间对齐，不减少IC或DO使用的完整标签截面，也不减少每轮训练日期。当前潜空间可包含没有收益标签但行情完整的股票；DO和IC只使用有效标签股票。

标签只来自训练窗口内且在重训截止日已到期的监督数据。训练期模板可以由整个训练集构造；这不是逐历史样本日的实盘模板回放。样本外推理仅依赖保存的生成器和截至预测日的40日特征，不读取记忆库、Critic或未来收益。

## 公共代码与产物

- `MarketStyleAlphaModel` 共用输入完整性检查、截面归一化入口、完整日期IC及无标签预测；LSG复用同一接口，保持其现有结构与训练目标。
- `models/adversarial.py` 提供梯度惩罚、Critic目标与冻结Critic的上下文；`data/latent_reference.py` 提供内存快照和无放回采样。
- 每轮记录G总损失、两项对抗损失、IC、Critic损失、GP、Wasserstein评分差、验证IC、记忆库版本及耗时；公共缓存日志报告加载/淘汰和Parquet打开增量。
- 权重：`data/model_workspace/dtan_alpha/artifacts/annual_YYYY/model.pkl`，为带版本的PyTorch checkpoint；同目录有manifest、loss_history和model_info。保存最优G及对应Critic权重供审查，预测不运行Critic。最终checkpoint不是优化器断点续训快照。
- 因子：`data/factors/stock/daily/YYYY/YYYYMMDD_ml.parquet` 的 `dtan_alpha` 列；沿用公共日期锁、原子合并写出和重复 `--config` 联合预测。
- tag包含 `fundamental`，防止含Barra基本面信息的模型被纯量价配置误选。

## CLI

在项目根目录，使用现有Python环境：

```powershell
& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\dtan_alpha.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\dtan_alpha.toml
# 或合并训练与预测：
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\dtan_alpha.toml
```

本轮仅实现代码和CPU定向验证，未调用fit、优化器更新、Slurm提交、正式预测写出或全量生成。

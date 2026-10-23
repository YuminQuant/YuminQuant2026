# Artifact Storage / 产物存储扩展

标准 Rust CLI 和 Python ML 入口默认使用本地文件系统，无需注册后端。部署入口可以在启动工作线程前注册一次可选后端，仅接管其 `handles(path)` 返回 true 的路径。

Standard Rust and Python ML entrypoints use local storage without backend registration. A deployment launcher may register one optional backend before starting workers; it only handles paths accepted by `handles(path)`.

## Rust 接口 / Rust interface

见 [`ArtifactBackend`](../src/artifact_io.rs)。接口提供存在性检查、目录创建、Parquet 表读写和锁；通过 `artifact_io::register(Arc<dyn ArtifactBackend>)` 注册，重复注册报错。

See `ArtifactBackend` for existence checks, directory creation, Parquet table IO and locking. Register it using `artifact_io::register(Arc<dyn ArtifactBackend>)`; duplicate registration fails.

后端必须自行实现安全发布和跨进程锁，尤其是日频宽表的读、合并、写操作。注册后端不等于接管所有文件访问；直接使用标准文件系统、输入加载器或流式写入器的路径需要单独确认支持范围。

Backends must implement safe publication and cross-process locking, particularly for read-merge-write operations on daily wide tables. Registration does not intercept all filesystem access: direct filesystem calls, input loaders and streaming writers require separate integration checks.

## Python 接口 / Python interface

见 [`output.artifact_io`](../../ml_alpha/yq_ml_alpha/output/artifact_io.py)。通过 `register(backend)` 注册，支持文件流、文本、Parquet、模型序列化及路径身份映射。后端负责原子提交；模型算法不需要依赖具体存储客户端。

The Python adapter supports file streams, text, Parquet, model serialization and stable path identities through `register(backend)`. The backend owns atomic publication; model algorithms need not depend on a particular storage client.

## 部署边界 / Deployment boundaries

通用仓库只保留接口，不存放私有客户端、凭证或企业路径。部署实现和配置应在仓库外维护。后端错误应明确返回，不能静默降级为另一个输出位置；节点清理与任务调度由部署层负责。

Keep private clients, credentials and organizational paths outside the public repository. Surface backend failures rather than silently redirecting outputs. Task scheduling and node cleanup belong to the deployment layer.

接入时应验证读写、并发合并、失败清理和本地默认行为，并将临时验证产物留在忽略目录。

Validate reads, writes, concurrent merges, failure cleanup and unchanged local defaults. Keep temporary validation artifacts in ignored directories.

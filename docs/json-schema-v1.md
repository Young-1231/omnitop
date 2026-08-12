# OmniTop JSON schema v1

`omnitop --json` 输出一个 UTF-8 JSON 对象。`--count N` 在 `N > 1` 时输出 JSON Lines，每行都是完整且独立的 v1 对象。

## 兼容策略

- `schema_version` 在破坏性变更时递增。
- 同一 schema 版本可以增加可选字段；消费者必须忽略未知字段。
- 缺失或驱动不支持的数值使用 `null`，不用 `NaN` 或 `Infinity`。
- 字节单位均为 bytes，速率均为 bytes/second，时间戳为 Unix seconds，百分比范围通常为 0–100；进程 CPU 可以超过 100，表示使用多个逻辑核。
- GPU `proc_map` 是内部聚合结构，不属于导出契约。

## 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | integer | 固定为 `1` |
| `version` | string | OmniTop 版本 |
| `sequence` | integer | 当前进程内单调递增的采样序号 |
| `time` | number | 本地快照 Unix 时间 |
| `generated_at` | string | UTC ISO 8601 时间 |
| `elapsed` | number | 与前一次采样之间的单调时间 |
| `sample_duration_ms` | number | 本次采集墙钟耗时，毫秒 |
| `hostname` | string | 主机名 |
| `uptime` | number | 主机运行秒数 |
| `logical_cpus` | integer | 逻辑 CPU 数 |
| `processes_total` | integer | 发现的进程总数；`--no-processes` 时为 0 |
| `warnings` | array[string] | 本次采集的降级与超时提示 |

## 资源对象

- `cpu`：`total`、`percpu`、`load`、`temp`、`freq`、`stats`、`history`。
- `memory`：`virtual`、`swap`、`history`。psutil namedtuple 会导出为具名对象。
- `disks.partitions[]`：设备、挂载点、文件系统、总量、已用、空闲和百分比。
- `disks.io[]`：设备读写速率、IOPS 和 busy 百分比。
- `network.interfaces[]`：链路状态、速率、包速率、错误和丢包计数。
- `gpu`：`available`、`degraded`、`error`、`driver`、`monitor_only`、`gpus[]`、`history`。
- `processes[]`：PID、安全启动 token、用户、状态、CPU、内存、I/O、GPU 设备/类型/显存和清洗后的命令行。

`gpu.monitor_only=true` 明确表示 NVML 连接不是 CUDA 计算上下文。真实占用来自各 GPU 对象的 `processes` 列表。

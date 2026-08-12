# OmniTop

OmniTop 是面向 Linux 训练机和多 GPU 工作站的终端监控器。它把 CPU、内存、磁盘、网络、进程与 NVIDIA GPU 状态放在同一个交互界面中，同时提供稳定的 JSON 快照接口，适合人工排障、脚本采集和长期运维。

当前版本：`2.0.0`。

## 主要能力

- 自适应终端布局：宽屏、普通终端和小于 80 列的窄终端都有独立布局。
- NVIDIA GPU：利用率、显存、温度、功耗、进程、compute/graphics 类型；详细视图额外采集时钟和 PCIe 吞吐。
- 进程视图：CPU、内存、I/O、GPU 显存、设备、命令行、受控环境变量和 PID 安全操作。
- 可靠速率：使用单调时钟计算速率，并用进程启动 token 防止 PID 复用污染缓存或误杀新进程。
- 故障降级：单个磁盘、网卡、GPU 或进程读取失败不会让整个界面退出；NVML 初始化失败会定期重试。
- 自动化接口：`--json` 输出带 schema 版本的快照，`--count N` 输出 JSON Lines。
- 低开销运行：数据变化或按键发生时才重绘；PCIe 和进程 I/O 等昂贵指标只在需要时采集。
- 运维诊断：`--diagnose` 检查 Python、依赖、终端和 NVML 状态。

## 安装

推荐使用项目自己的虚拟环境：

```bash
cd /data/huangfy/projects/omnitop
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

安装后可直接运行：

```bash
/data/huangfy/projects/omnitop/.venv/bin/omnitop
```

本机还保留兼容入口：

```bash
python3 /data/huangfy/omnitop.py
```

## 常用方式

交互监控：

```bash
omnitop
omnitop --user huangfy
omnitop --gpu-only --sort gpu
omnitop --full --interval 2
```

一次性终端快照：

```bash
omnitop --once --rows 20
omnitop --once --no-gpu --no-color
```

机器可读输出：

```bash
omnitop --json
omnitop --json --pretty
omnitop --json --count 10 --interval 2 > snapshots.jsonl
omnitop --json --gpu-only | jq '.gpu.gpus, .processes'
```

快速诊断：

```bash
omnitop --diagnose
omnitop --diagnose --json --pretty
omnitop --debug-log /tmp/omnitop.log
```

如果标准输入或标准输出不是 TTY，OmniTop 会自动生成一次快照并退出，不会像交互程序那样永久等待按键。需要纯数据时仍建议显式使用 `--json`。

## 交互按键

| 按键 | 操作 |
| --- | --- |
| `↑` / `↓`, `PgUp` / `PgDn`, `Home` / `End` | 移动进程选择 |
| `c` / `m` / `g` / `i` / `p` / `n` | 按 CPU、内存、GPU、I/O、PID、名称排序 |
| `/` | 输入进程过滤条件 |
| `u` | 清空过滤条件 |
| `d` | 切换摘要/详细资源视图 |
| `v` | 显示所选进程详情 |
| `e` | 显示所选进程的白名单 ML/GPU 环境变量 |
| `1` | 切换逐核 CPU 信息 |
| `a` | 切换全部网卡和挂载点 |
| `Space` | 暂停/恢复采样 |
| `+` / `-` | 调整采样间隔 |
| `r` | 重置速率和趋势历史 |
| `k` | 请求终止所选进程；再次确认后才发送信号 |
| `h` / `?` | 帮助 |
| `q` | 退出 |

## GPU 占用语义

OmniTop 使用 NVML 读取 GPU。NVML 在监控器存活期间可能打开 `/dev/nvidia*` 文件描述符，这只表示监控连接，不代表 OmniTop 建立了 CUDA 计算上下文，也不代表它占用了 GPU 显存。

判断真实 GPU 进程时，OmniTop只采用 NVML 的 compute/graphics running-process 查询。排障时也应以 `nvidia-smi` 的进程表和 `nvidia-smi pmon` 为准，不能把 `lsof`/`fuser` 看到设备文件句柄直接等同于计算占用。

同一 PID 同时出现在 compute 和 graphics 查询中时，OmniTop 会合并类型并按每块设备的最大显存值计算，避免重复累加。

## 性能建议

- 默认摘要视图不采集昂贵的逐进程 I/O、GPU PCIe、时钟、编码器和解码器指标；详细视图中的 GPU 慢指标最多缓存 5 秒。
- `--full` 或按 I/O 排序时会启用更多采集，开销会相应增加。
- 进程数很多的服务器可使用 `--user USER`、`--gpu-only` 或 `--no-processes` 限定范围。
- 如果一次采集时间超过刷新间隔，顶部会显示 `DEGRADED`，JSON 的 `warnings` 也会记录该情况。
- 不建议把间隔设置到低于主机实际采集耗时；允许范围为 0.2–60 秒。

## JSON 契约

每个快照都包含：

- `schema_version`：当前为 `1`；不兼容字段变更时才递增。
- `version`：生成快照的 OmniTop 版本。
- `generated_at`：UTC ISO 8601 时间。
- `sample_duration_ms` 与 `warnings`：采集耗时和降级信息。
- `cpu`、`memory`、`disks`、`network`、`gpu`、`processes`：与交互界面共用的采集结果。

详细字段和兼容规则见 [docs/json-schema-v1.md](docs/json-schema-v1.md)。

## 安全边界

- `k` 操作默认只提出请求；`y` 发送 `SIGTERM`，大写 `K` 才发送 `SIGKILL`。
- PID 1 和 OmniTop 自身永远不会被信号操作。
- 确认前后会重新比较进程启动 token；PID 已被复用时拒绝操作。
- 环境变量详情只显示代码中的 ML/GPU 白名单，不导出完整进程环境。
- JSON 和终端文本会清除控制字符，避免进程命令行注入终端转义序列。

## 开发与验证

```bash
cd /data/huangfy/projects/omnitop
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest --cov=omnitop --cov-report=term-missing
```

源码入口是 `src/omnitop/app.py`，控制台入口是 `omnitop.app:main`。大数据集、模型权重和运行快照不应写入源码目录。

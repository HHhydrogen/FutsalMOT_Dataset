# 可复现性与数据集 Manifest

本文档说明 GRF-UE 桥接的**随机种子体系**（seed 可复现性）与**数据集级 manifest**
（索引、校验和、去重检测、稳定 fingerprint）。

## 一、Seed：让 episode 真正可复现

### 1. root seed 与子 seed

GRF 引擎只接受一个 `game_engine_random_seed`。从用户提供的 **root seed** 通过
SHA-256 派生四个命名空间子 seed：

| namespace | 用途 |
|-----------|------|
| `grf_game_engine` | **真正传入 GRF** 的 `game_engine_random_seed` |
| `python` | Python 标准库 `random.seed` |
| `numpy` | `numpy.random.seed` |
| `ue_visual` | 预留：未来 UE 视觉随机化用（本轮仅记录） |

派生算法（`src/grf_ue_bridge/seeds.py`）：

```python
SEED_POLICY = "futsalmot_seed_v1"
payload = f"{SEED_POLICY}:{root_seed}:{namespace}".encode("utf-8")
sub_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
```

- 相同 root_seed + namespace ⇒ 相同子 seed（跨进程、跨机器一致）；
- 不使用 Python 内置 `hash()`（其跨进程/平台不稳定，不适合持久协议）；
- 改变派生算法时必须换新 policy 名（如 `futsalmot_seed_v2`），**不得静默改变 v1**。

### 2. 真正传入 GRF

`create_env()` 把 `game_engine_random_seed` 放进 `other_config_options`：

```python
other_config_options = {**user_options, "action_set": "v2",
                        "game_engine_random_seed": seeds.grf_game_engine_seed}
```

gfootball 的 `scenario_builder` 只有在 **未提供** `game_engine_random_seed` 时
才用 `random.randint(0, 2e9)` 兜底（即旧版本不可复现）；提供后写入
`scenario_cfg.game_engine_random_seed`，轨迹确定。同时设置 Python 与 NumPy seed。

### 3. metadata

`meta.json` 新增（旧 episode 无此字段，validator 标记为 legacy）：

```json
"randomness": {
  "policy": "futsalmot_seed_v1",
  "root_seed": 1001,
  "grf_game_engine_seed": 1800042844,
  "python_seed": 1617145048,
  "numpy_seed": 695028763,
  "ue_visual_seed": 850854685
}
```

`source.seed` 保留并须等于 `randomness.root_seed`。validator 检查：policy 非空、
seed 为合法整数、子 seed 与按当前 policy 派生结果一致；无 randomness 的旧 episode
仍可读取，仅提示 `legacy seed metadata`。

同时，导出时把**实际使用的配置快照**写入 `episode_root/provenance/`：

```text
provenance/export_config.json          # 实际使用的 ExportConfig（含 seed）
provenance/external_sources.lock.json  # 外部仓库锁定提交号
```

供 manifest 做确定性 provenance。

### 4. 如何复现

```powershell
uv run grf-ue export --config configs/export/standard_300steps_10fps.json --output outputs/ep_s1001 --seed 1001
```

CLI `--seed` 优先级：**CLI > 配置文件 > 默认值**。运行时打印 root / GRF engine / policy。
相同 seed 两次导出，`frames.jsonl` 的 SHA-256 完全一致（见集成测试）。

### 5. 复现边界

> 在**相同项目 commit、GRF commit、配置、Python 环境和操作系统类别**下，episode
> 轨迹可复现。不同 GRF、编译器、平台或依赖版本之间**不保证**二进制级一致。

### 6. 比较 trajectory hash

`frames.jsonl` 的原始字节 SHA-256 即 `trajectory_hash`（见 manifest）。两个 episode
若 hash 相同 ⇒ 轨迹完全一致。

## 二、Dataset Manifest

### 1. 用途

`grf-ue build-manifest` 为**明确指定的 episode** 生成数据集根目录级索引：

```text
dataset_root/
├── dataset_manifest.json       # 汇总索引 + fingerprint
├── checksums/
│   └── episode_0001.jsonl      # 逐文件校验和（path, size, sha256）
└── episode_0001/               # 各 episode（需自包含 meta.json/frames.jsonl）
```

它只做只读汇总与校验，**不**负责生成 episode、调用 UE、重试、划分训练集、
自动运行 validator。

### 2. checksum profile

| profile | 校验范围 |
|---------|---------|
| `metadata` | meta/frames/render_summary + 每相机元数据（camera/annotations/mask_config/seqinfo/gt）+ provenance |
| `final`（默认） | metadata + `img1/` + `mask/` + `labels/det/` + `labels/seg/` |
| `all` | final + `render/` + `render_mask/`（原始 EXR 体积大，非默认） |

### 3. 命令

```powershell
# 构建
uv run grf-ue build-manifest G:/FutsalMOT_Dataset `
  --episode episode_0001 `
  --dataset-id futsalmot_soak_v001 `
  --checksum-profile final `
  --workers 4

# 校验（0=通过；1=内容不匹配/缺失；2=manifest/schema/参数错误）
uv run grf-ue verify-manifest G:/FutsalMOT_Dataset
```

`--episode` 可重复指定；缺省只纳入满足合法 episode 结构（含 camera.json）的目录。

### 4. dataset fingerprint

只由稳定内容计算：

```json
{"schema": "futsalmot_dataset_fingerprint_v1", "episodes": [
  {"episode_id": "...", "relative_path": "...", "trajectory_hash": "...",
   "checksums_file_sha256": "...", "config_hashes": {...}}]}
```

episode 按 `episode_id`/`relative_path` 排序、固定 separators。**不含** `created_at_utc`、
绝对路径、机器名、用户名、耗时。数据集整体移动后 fingerprint 不变；内容不变重复构建
不变。

### 5. 重复轨迹检测

- `duplicate_seed_groups`：按 `(root_seed, scenario, export_config_hash)` 分组，重复即警告
  「possible duplicate seed/config combination」（有意固定轨迹时可忽略）。
- `duplicate_trajectory_groups`：按 `trajectory_hash` 分组，重复即「duplicate trajectory」。
- 不同 root seed 产生相同 trajectory → `manifest.warnings` 提示「可能的 seed 传播失败」。
- 默认只警告不阻止；`--strict-duplicates` 可使构建非零退出。

### 6. 路径可移植性

manifest 与 checksum 文件全部使用 **POSIX 相对路径**，无盘符/反斜杠/绝对路径。
`verify-manifest` 会拒绝逃逸 dataset root 的路径（如 `../../outside.txt`）。

### 7. 旧 episode 无 provenance 的限制

旧 episode（本轮 soak 之前导出）没有 `randomness` 与 `provenance/`：

- `root_seed` 回退读 `source.seed`（best-effort）；
- `grf_game_engine_seed` / `seed_policy` / `config_hashes` 为 null/空 —— **不伪造**
  provenance，也不声称已按当前 policy 可复现；
- 原始 `render/`、`render_mask/` 在 `final` profile 下不在校验范围，verify 列为
  extra 警告（`--strict` 才失败）；需要校验原始帧用 `--checksum-profile all`。

## 三、真实 soak 数据集实测（episode_0001，300 步 × 4 相机）

| 指标 | 值 |
|------|-----|
| dataset_id | `futsalmot_soak_v001` |
| episode 数 | 1（episode_0001） |
| 相机 | 4 |
| camera-frame | 1200（img1=1200、mask=1200、annotation=1200、det=1200、seg=1200） |
| checksum 文件行数 | 4823 |
| 总字节（final 校验范围） | 4.81 GB rgb + 29 MB mask + 21 MB annotations + 10 MB labels |
| 原始帧 | raw_rgb=3600、raw_object_id_exr=3600 |
| trajectory_hash | `17d301f76b19031862fc61b456f3d686…` |
| dataset_fingerprint | `1b62e41361a855411a4bbe6ccd4b352e188b77c834b074d8079d86b8c4698c62` |
| build 耗时 / 峰值 RSS | 44.7 s / 99 MB（workers=4, final） |
| verify 耗时 / 峰值 RSS | 9.0 s / 101 MB（workers=4） |

> 该 episode 为 legacy（旧代码导出，无 randomness/provenance）：`root_seed=42`、
> GRF seed 未知。用新代码重新导出会因 seed 派生改变轨迹，需同步重渲重标注——因此
> 本轮保留旧轨迹并如实标记 legacy。

## 四、推荐命令汇总

```powershell
# 1) 可复现导出（seed 覆盖）
uv run grf-ue export --config configs/export/standard_300steps_10fps.json --output outputs/ep_s1001 --seed 1001

# 2) 构建 manifest
uv run grf-ue build-manifest G:/FutsalMOT_Dataset --episode ep_s1001 --episode ep_s1002 --dataset-id futsalmot_v001 --checksum-profile final

# 3) 校验 manifest
uv run grf-ue verify-manifest G:/FutsalMOT_Dataset
```

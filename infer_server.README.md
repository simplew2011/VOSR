# infer_server.py — VOSR2 OpenAI 兼容常驻超分服务

基于 `inference_vosr_onestep_pipeline.py` 改造的常驻 GPU 视频超分 HTTP 服务，
采用 **OpenAI 兼容风格** API（`/v1/*`、Bearer 鉴权、统一响应/错误信封）。

- 启动时加载一次 VOSR2（LightningDiT 1.4B + Qwen-Image 2D VAE + DINOv2-L）并常驻显存，
  重复任务不重新加载权重。
- **同时支持图片和视频超分**：按输入扩展名自动识别；图片输出 PNG，视频输出 MP4。
- `POST /v1/files` 上传（或用服务器本地路径），`POST /v1/upscales`
  异步排队、串行推理。
- 支持任务状态查询、取消、下载超分结果。
- 视频可用 **帧级多卡并行**（`num_gpus`）：N 张卡各跑一段连续帧，再拼接；实测 2 卡约 1.4–1.7x、4 卡约 2.2–2.9x（视视频长度/分辨率而定，详见「多卡并行」）。
- 视频输出用 ffmpeg 编码（libx264 / h264_nvenc）并**保留原视频音轨**，保留原 FPS。
- 逐帧独立超分（VOSR2模型原生无时序模块）：**输入 N 帧 → 输出 N 帧**。
- 视频超分支持多图batch推理

## 启动

```bash
cd /cx8k/fs101/wzp/code/0929/VOSR
conda activate vosr

# 可选：设置 API Key 启用 Bearer 鉴权
export VOSR_API_KEY=sk-vosr-demo

# 单卡
CUDA_VISIBLE_DEVICES=3 python infer_server.py

# 多卡（视频帧级并行）：暴露 4 张卡，并把默认帧级并行数设为 4
CUDA_VISIBLE_DEVICES=3,4,5,6 VOSR_NUM_GPUS=4 python infer_server.py

# 用命令行参数（会覆盖同名 VOSR_* 环境变量）
python infer_server.py --port 18303 --num-gpus 4 --dtype bf16 --api-key sk-vosr-demo
python infer_server.py --help        # 查看全部参数
```

> **参数优先级：命令行参数 > 环境变量 > 内置默认值。** 每个命令行参数都有一个对应的
> `VOSR_*` 环境变量作为默认值；两者都不给则用内置默认。启动时会打印一行 `config: ...`
> 汇总最终生效的配置。


首次启动加载模型（本机约 50s），看到 `runtime[0] ready: load_seconds=...` 与
`Application startup complete` 后即可接受请求（`/health` 的 `resident` 变为 `true`）。

### 依赖

```bash
pip install fastapi uvicorn python-multipart
```

> 若共享 conda 环境的 `site-packages` 不可写（属主非你），用 `pip install --user ...` 装到
> `~/.local`（每台机器各自的 home，不随 NFS 环境共享，需在各机器各装一次）。

另需系统 `ffmpeg`/`ffprobe` 用于编码与音轨 remux。

### 环境兼容：坏掉的 sageattention 不再阻塞启动

共享/迁移过的 conda 环境若带有**为其它 CUDA 运行时编译**的 `sageattention`（例如链接
`libcudart.so.13`），本机 `import sageattention` 会失败；而 `diffusers` 会在 import 时**急切地**
导入它，导致 `import diffusers` 乃至整个服务崩溃：

```
ImportError: libcudart.so.13: cannot open shared object file
```

**已在仓库侧自动兜底**（`inference_vosr_onestep.py` 顶部的 `_ensure_sageattention_importable()`）：
真实 `import sageattention` 失败时，注册一个**无害 stub**，让 `diffusers` 正常导入；VOSR 用的是
默认 PyTorch SDPA 注意力，stub 的函数不会被调用（万一被调用会抛出明确错误）。启动日志会提示：

```
sageattention failed to import (broken env build); using a stub so diffusers imports. VOSR attention falls back to PyTorch SDPA.
```

要彻底修复（可选）：在能用的 CUDA 运行时下重装匹配的 `sageattention`，或 `pip install --user`
一个可用版本覆盖 env 里的坏版本。无需为跑服务而处理它。

### 配置（命令行参数 / 环境变量）

每个配置项都有等价的**命令行参数**与 **`VOSR_*` 环境变量**，优先级：
**命令行参数 > 环境变量 > 内置默认值**（都不给则用默认）。`python infer_server.py --help` 可查看全部。

| 命令行参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--host` | `VOSR_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `VOSR_PORT` | `18303` | 监听端口 |
| `--api-key` | `VOSR_API_KEY` | 空 | **设置后启用 Bearer 鉴权**；为空则不鉴权 |
| `--ckpt` | `VOSR_CKPT` | `./preset/ckpts/VOSR2` | 模型目录（含 `args.json` + `checkpoints/`） |
| `--work-dir` | `VOSR_WORK_DIR` | `~/.vosr_server` | 输入/任务工作目录（需可写） |
| `--output-dir` | `VOSR_OUTPUT_DIR` | `<work-dir>/outputs` | 超分结果输出目录 |
| `--encoder` | `VOSR_ENCODER` | `auto` | 编码器：`auto`/`libx264`/`nvenc`（请求体可覆盖） |
| `--dtype` | `VOSR_DTYPE` | `bf16` | DiT/DINOv2 精度：`bf16`/`fp32`（VAE 恒 fp32） |
| `--tile-size` | `VOSR_TILE_SIZE` | `512` | 默认 DiT 分块大小（质量关键项） |
| `--vae-tile-size` | `VOSR_VAE_TILE_SIZE` | `0` | 默认 VAE 分块（0/None=全图，更快；仅超大图 OOM 时设 >0） |
| `--batch-size` | `VOSR_BATCH_SIZE` | `8` | 默认每批帧数（视频） |
| `--seed` | `VOSR_SEED` | `42` | 默认随机种子（请求字段 `seed` 可覆盖） |
| `--num-gpus` | `VOSR_NUM_GPUS` | `1` | 视频请求字段 `num_gpus` 的**默认值**（请求可覆盖）；**非常驻副本**，详见「多卡并行」 |
| `--max-queue` | `VOSR_MAX_QUEUE` | `8` | 最大排队任务数 |
| `--model-name` | `VOSR_MODEL_NAME` | `VOSR2-1.4B-OneStep` | `/v1/models` 报告的模型名 |
| `--log-level` | `VOSR_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

示例：

```bash
# 命令行参数（覆盖同名环境变量）
python infer_server.py --port 18303 --num-gpus 4 --dtype bf16 --api-key sk-vosr-demo

# 等价的环境变量写法
VOSR_PORT=18303 VOSR_NUM_GPUS=4 VOSR_DTYPE=bf16 VOSR_API_KEY=sk-vosr-demo python infer_server.py
```

启动日志会打印最终生效配置：

```
... | INFO | vosr | config: host=0.0.0.0 port=18303 ... num_gpus=4 encoder=auto auth=True
```

## 多卡并行（`num_gpus`）

### 参数语义（重要）

- `--num-gpus` / `VOSR_NUM_GPUS` **只设置「视频请求 `num_gpus` 的默认值」**，
  **不会在启动时为每张卡常驻一份模型副本**。无论它设多大，启动时都**只在首张可见 GPU 上常驻一份模型**。
- 请求体里的 `num_gpus` 优先级最高，覆盖启动默认值。
- **实际生效卡数** = `min(请求 num_gpus（或启动默认）, 可见卡数)`。上限是**可见卡数**
  （即 `CUDA_VISIBLE_DEVICES` 列出的卡数），**不是 `--num-gpus`**：
  - 用 `CUDA_VISIBLE_DEVICES=3` 限成单卡 → 请求再写 `num_gpus>1` 也只会有 1 卡（被截断）。
  - 只要卡可见，即使 `--num-gpus 1`，请求仍可用 `num_gpus>1` 覆盖为多卡。
- **图片忽略 `num_gpus`**（单帧，始终单卡）。

### 多卡如何工作

`num_gpus>1` 的视频走**帧级数据并行**，**不重载模型**：

- **常驻模型永远保持加载**（进程内、GPU0），它**就是 GPU0 分片的执行者**（不会再为 GPU0 另起副本 → 不会双份显存）。
- **GPU1..N-1 用常驻 worker 池**：每张卡一个长驻 worker 进程，**模型只加载一次**，之后跨请求复用。
- 请求开始时按当前 `num_gpus` 确保池里有 N-1 个 worker（**num_gpus 不变 → 完全复用、零加载**；
  变大 → 只新起缺的那几个；变小 → 复用已有子集，多余的先留着）。
- 每个请求结束后**只释放激活、保留权重**（回到"干净加载态"），所以下一个请求立即开跑。

实现放在 `vosr_multigpu.py`，HTTP 服务与 CLI（`--num_gpus`）共用：

- 分片 worker 是普通函数 `run_video_segment(cfg, progress_cb, models=None)`；入口 `python -m vosr_multigpu --server <base.json>`（常驻模式，读 stdin 的 JSON 作业）或 `python -m vosr_multigpu <cfg.json>`（CLI 一次性模式）。
- 主进程用 **JSON-lines 协议**读 worker stdout 汇总进度；worker 的日志原样转发——**不解析 tqdm/CLI 字符串**。
- 每张卡仍是**独立进程**：干净 CUDA 上下文、故障隔离（worker 崩不影响服务）。进程退出（或 stdin EOF，如服务被 kill）即回收该卡显存。

> **权衡**：常驻 worker 会一直占着各卡的模型显存（每卡 ~4GB），这是"请求间免重载"的代价。
> 不常做多卡时，可让它按空闲回收（待实现）。

### 影响与代价

- **首次多卡 / 每次 `num_gpus` 变大**：需要为新启的卡各加载一次模型（~17s/卡）；
  **`num_gpus` 不变时零加载**（worker 复用），常驻模型也始终在位。
- **常驻占用**：常驻模型（GPU0）+ 每个常驻 worker（各卡）会一直占着各自卡的模型显存（每卡 ~4GB），
  这是"请求间免重载"的代价。缩小 `num_gpus` 后多余的 worker 会保留（暂不自动回收）。
- **单卡串行队列**：多卡请求同样一次只跑一个任务（不并发多个任务）。
- **取消只中止当前任务，不卸载模型**：取消时各分片在当前 batch 边界停止（常驻 worker 收到 `cancel` 命令后中止当前作业并保留权重），
  常驻模型与 worker 都保持加载；下一个请求直接复用，无需重载、不留孤儿。
- 共享机器上，其他进程占用某张卡会导致该 shard 显存不足 → **整个任务失败**（也可用 `vae_tile_size`/`batch_size` 降峰值，见下节）。
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已默认设置**（服务启动时 `setdefault`，worker 继承）：
  否则即使总显存够，PyTorch 分配器碎片也可能触发 "reserved but unallocated" 型 OOM。
- **分片失败即快速失败**：任一 shard 报错立刻终止其余 shard，不会等健康 shard 跑完（否则 OOM 会拖延数分钟才上报）。
- **失败会回传真实原因**：只要有一个 shard 失败，整个任务 `failed`，且 `error.message` 携带子进程最后
  的错误输出（如 `torch.OutOfMemoryError: CUDA out of memory...` 的完整栈），不会再出现「see its log above」这种查不到原因的提示。
- **不做 bicubic 回退**：分片里某批帧失败时**直接报错**，不会静默用 bicubic 顶替（否则会以 `completed` 返回非 VOSR 结果）。

### 实测（H800 80GB，单卡常驻）

720p→1440p，5s（152 帧），`tile_size=512, vae_tile_size=0, batch_size=8`：

| `num_gpus` | 处理耗时 `elapsed_seconds` | 加速比 |
|---|---|---|
| 1 | 191.4s | 1.00× |
| 2 | 132.0s | 1.45× |
| 4 | 87.2s | 2.19× |

1080p→4K，5s（126 帧），`tile_size=512, vae_tile_size=1024, batch_size=8`：

| `num_gpus` | 处理耗时 `elapsed_seconds` | 加速比 |
|---|---|---|
| 1 | 395–396s | 1.00× |
| 2 | 229–280s | 1.4–1.7×（波动较大） |
| 4 | 134–135s | ~2.93× |

> 对比时用任务对象的 **`elapsed_seconds`（纯处理耗时）**，不要用 `total_seconds`（含排队时间）。

### 怎么选

- 长视频 / 追吞吐：把可见卡数拉满（`CUDA_VISIBLE_DEVICES=... VOSR_NUM_GPUS=N`，或请求里指定）。
- 短视频：单卡或少量卡即可，多卡重载开销可能吃掉收益。
- 想要「真单卡」、禁止请求改多卡：启动时用 `CUDA_VISIBLE_DEVICES=<单卡>` 限卡。

## 鉴权

- 设置 `VOSR_API_KEY` 后，所有 `/v1/*` 请求必须带 `Authorization: Bearer <VOSR_API_KEY>`。
- 未设置时不鉴权（本地开发友好）。`/health` 永不鉴权，其 `auth` 字段表示鉴权开关。
- 鉴权失败返回 `401`，错误体为 OpenAI 风格。

## API

Base URL：`http://127.0.0.1:18303`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（无需鉴权） |
| GET | `/v1/models` | 列出可用模型 |
| GET | `/v1/models/{model}` | 查询单个模型 |
| POST | `/v1/files` | 上传图片/视频，返回 `file_id` |
| GET | `/v1/files` | 列出已上传文件 |
| DELETE | `/v1/files/{file_id}` | 删除已上传文件 |
| POST | `/v1/upscales` | 提交超分任务（图片或视频，`202`） |
| GET | `/v1/upscales` | 任务列表（最近 10 个） |
| GET | `/v1/upscales/{id}` | 任务状态 |
| POST | `/v1/upscales/{id}/cancel` | 取消任务 |
| GET | `/v1/upscales/{id}/content` | 下载超分结果（PNG/MP4） |

### 各端点示例

下文统一约定（`AUTH` 仅在设置了 `VOSR_API_KEY` 时需要，否则删掉；所有请求用 `curl -sS`）：

```bash
BASE=http://127.0.0.1:18303
AUTH="Authorization: Bearer sk-vosr-demo"   # 未设置 API_KEY 时删掉
```

**GET `/health`**（无需鉴权）
```bash
unset http_proxy https_proxy

curl -sS $BASE/health | python -m json.tool
# 关心就绪状态：resident 为 true 才能接请求
curl -sS $BASE/health | python -c 'import json,sys;d=json.load(sys.stdin);print("resident",d["resident"],"active",d["active"],"queued",d["queued"])'
```

**GET `/v1/models`** / **GET `/v1/models/{model}`**
```bash
curl -sS -H "$AUTH" $BASE/v1/models | python -m json.tool
curl -sS -H "$AUTH" $BASE/v1/models/VOSR2-1.4B-OneStep | python -m json.tool
curl -sS -H "$AUTH" $BASE/v1/models/nope   # 404 model_not_found
```

**POST `/v1/files`**（multipart，字段名 `file`）
```bash
UPLOAD=$(curl -sS -X POST -H "$AUTH" -F 'file=@preset/datasets/videos/bg_video.mp4' $BASE/v1/files)
FILE_ID=$(echo "$UPLOAD" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "FILE_ID=$FILE_ID"
```

**GET `/v1/files`**
```bash
curl -sS -H "$AUTH" $BASE/v1/files | python -m json.tool
```

**DELETE `/v1/files/{file_id}`**
```bash
curl -sS -X DELETE -H "$AUTH" $BASE/v1/files/$FILE_ID
# 该文件正被未完成任务引用 -> 409 file_in_use
# 文件已不存在（例如其任务已结束被自动清理）-> 404 file_not_found
```
> 上传的文件会在**引用它的任务全部到达终态后被自动清理**（释放磁盘），所以任务跑完后
> 再 `DELETE` 通常得到 `404`。

**POST `/v1/upscales`**（提交，`202`；图片/视频同一入口，`file_id` 与 `source` 二选一）
```bash
# 用已上传的 file_id
curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"file_id":"'"$FILE_ID"'","scale":2,"tile_size":512,"batch_size":8}' \
  $BASE/v1/upscales | python -m json.tool
# 用服务器本地路径（图片或视频）
curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"preset/datasets/videos/bg_video.mp4","scale":2,"tile_size":512,"batch_size":8}' \
  $BASE/v1/upscales | python -m json.tool
```

**GET `/v1/upscales`**（最近 10 个任务）
```bash
curl -sS -H "$AUTH" $BASE/v1/upscales | python -m json.tool
```

**GET `/v1/upscales/{id}`**（状态）
```bash
UPSCALE_ID=upscale_..._2x
curl -sS -H "$AUTH" $BASE/v1/upscales/$UPSCALE_ID | python -m json.tool
```

**POST `/v1/upscales/{id}/cancel`**
```bash
curl -sS -X POST -H "$AUTH" $BASE/v1/upscales/$UPSCALE_ID/cancel | python -m json.tool
```

**GET `/v1/upscales/{id}/content`**（下载；未完成返回 `409`）
```bash
curl -sSL -H "$AUTH" $BASE/v1/upscales/$UPSCALE_ID/content -o /tmp/output.png   # 图片 -> PNG
curl -sSL -H "$AUTH" $BASE/v1/upscales/$UPSCALE_ID/content -o /tmp/output.mp4   # 视频 -> MP4
```

### 快速开始：上传 → 超分 → 下载

> `AUTH` **只在设置了 `VOSR_API_KEY` 时才需要**，否则删掉该变量（带上会返回 401）。
> 提交/轮询一律用 `curl -sS`，**不要加 `-f`**：`-f` 会在 4xx/5xx 时静默丢弃响应体，
> 让 `python -c ...["id"]` 报 `Expecting value`，看起来像"服务没反应"。

```bash
BASE=http://10.24.9.24:18303
AUTH="Authorization: Bearer sk-vosr-demo"   # 未设置 API_KEY 时删掉此变量

# 1. 上传视频，得到 file_id
UPLOAD=$(curl -sS -X POST -H "$AUTH" \
  -F 'file=@preset/datasets/videos/bg_video.mp4' \
  $BASE/v1/files)
FILE_ID=$(echo "$UPLOAD" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "FILE_ID=$FILE_ID"

# 2. 提交超分任务（秒级返回 202，后台排队执行）
JOB=$(curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"file_id":"'"$FILE_ID"'","scale":2,"tile_size":512,"vae_tile_size":0,"batch_size":8}' \
  $BASE/v1/upscales)
UPSCALE_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "UPSCALE_ID=$UPSCALE_ID"

# 2.1 或直接上传本地路径及发送任务
JOB=$(curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"preset/datasets/videos/bg_video.mp4","scale":2,"tile_size":512,"vae_tile_size":0,"batch_size":8}' \
  $BASE/v1/upscales)
UPSCALE_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "UPSCALE_ID=$UPSCALE_ID"


JOB=$(curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"1789638866b741.png","scale":2,"tile_size":512,"vae_tile_size":1024,"batch_size":8}' \
  $BASE/v1/upscales)
UPSCALE_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "UPSCALE_ID=$UPSCALE_ID"

# 3. 轮询状态（json.tool 美化输出；822 帧 2x 约 15 分钟，属正常）
watch -n 2 "curl -sS -H \"$AUTH\" $BASE/v1/upscales/$UPSCALE_ID | python -m json.tool"

# 4. 下载结果（-o 指向可写目录；仓库目录可能不可写，见"常见问题"）
curl -sSL -H "$AUTH" "$BASE/v1/upscales/$UPSCALE_ID/content" -o /tmp/output_upscaled1.mp4

# 取消任务（可选）
curl -sS -X POST -H "$AUTH" $BASE/v1/upscales/$UPSCALE_ID/cancel | python -m json.tool
```


### 图片超分 / 多卡视频

```bash
# 图片（PNG 输出）
curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"preset/datasets/inp_data/Nikon_024_LR4.png","scale":4,"tile_size":512}' $BASE/v1/upscales | python -m json.tool

# 视频，2 卡帧级并行（可选：覆盖启动时的默认；不写则用 --num-gpus/VOSR_NUM_GPUS 的默认值）
curl -sS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"source":"preset/datasets/videos/36s.mp4","scale":2,"num_gpus":2,"tile_size":512,"batch_size":8}' \
  $BASE/v1/upscales  | python -m json.tool
```

### 提交任务 body

```json
{
  "file_id": "file_...",     // 与 source 二选一
  "source": null,            // 服务器本地图片或视频路径（与 file_id 二选一）
  "model": null,             // 可选，OpenAI 兼容字段；填了须等于模型名
  "scale": 2.0,              // 1.0 ~ 8.0
  "seed": 42,                // 可选，随机种子；不填则用服务端 VOSR_SEED（默认 42）
  "tile_size": 512,          // 可选，覆盖默认；QUALITY 关键项（512≈训练分辨率）
  "vae_tile_size": 0,        // 可选，0/None=全图更快（多 batch 视频也推荐）；仅高分辨率大 batch OOM 时设 1024
  "batch_size": 8,           // 可选，每批帧数（视频）；显存随 batch×分辨率增长
  "num_gpus": 1,             // 可选，视频帧级多卡数；图片忽略
  "align_method": "wavelet", // wavelet / adain / nofix
  "encoder": "auto"          // auto / libx264 / nvenc（视频）
}
```

- **图片**（`.png/.jpg/.jpeg/.bmp/.tif/.tiff/.webp`）→ 输出 PNG（`image.upscale`）。
- **视频**（`.mp4/.mov/.mkv/.avi/.webm/...`）→ 输出 MP4（`video.upscale`）。
- `num_gpus > 1` 时视频按帧分片到多张卡（复用 `inference_vosr_onestep_pipeline.py --num_gpus`），
  完成度高但需 ffmpeg 拼接；`1` 时用常驻单卡路径（最快启动、无重载）。
- 未提供 `file_id` 或 `source` 之一、或同时提供，返回 `400`。

### 任务对象

```json
{
  "id": "upscale_6f83e5...4d2a_9f3c1a_2x",
  "object": "video.upscale",
  "model": "VOSR2-1.4B-OneStep",
  "file_id": "file_6f83e5...4d2a",
  "status": "queued",
  "phase": "waiting",
  "progress": 0.0,
  "created_at": 1789546909,
  "started_at": null,
  "finished_at": null,
  "queue_seconds": null,
  "elapsed_seconds": null,
  "total_seconds": 1.2,
  "queue_position": 1,
  "request": {
    "kind": "video", "scale": 2.0, "seed": 42, "tile_size": 512, "vae_tile_size": 0,
    "batch_size": 8, "align_method": "wavelet", "encoder": "auto", "num_gpus": 1,
    "source_name": "bg_video.mp4",
    "source_width": 1280, "source_height": 720,
    "source_frames": 822, "source_fps": 29.964, "source_duration_seconds": 27.433,
    "target_width": 2560, "target_height": 1440
  },
  "result": null,
  "error": null
}
```

- `status`：`queued | running | completed | failed | cancelled`
- `phase`：`waiting | preparing | upscaling | encoding | audio_remux | completed | failed | cancelled`
- `progress`：0~100 的百分比；`queue_position` 仅在 `queued` 时出现。
- `request` 含输入探测信息：`source_name`、`source_width/height`、`source_frames`、
  **`source_fps`**、**`source_duration_seconds`**（视频；图片为 `0.0`/`null`），以及
  `target_width/height`。`status` 轮询时即可看到输入时长与帧率。
- 各时间戳为 **Unix 秒**（OpenAI 风格）。
- 耗时字段（秒，均为数字或 `null`）：
  - `queue_seconds`：排队等待耗时（`started_at - created_at`；尚未开始为 `null`）。
  - `elapsed_seconds`：处理耗时；`running` 时按 `started_at` 实时递增，**成功**完成后等于
    `result.elapsed_seconds`（纯推理/编码时长），失败/取消则为 `finished_at - started_at`。
    队列中为 `null`。
  - `total_seconds`：从创建到完成（`finished_at - created_at`）的墙钟耗时；
    未完成时按当前时间实时递增。
- 完成时 `result`：

  ```json
  {"status":"completed","scale":2.0,"width":2560,"height":1440,"frames":822,
   "fps":30.0,"elapsed_seconds":1010.0,"encoder":"auto","resident":true,
   "engine_load_seconds":17.6,"timings":{"inference":995.0,"encode":12.1,"audio_remux":1.3}}
  ```

- 失败时 `error`：`{"message":"...","type":"inference_error","param":null,"code":null}`

### 任务 ID 与输入对齐

```
upscale_{输入指纹32hex}_{6位随机hex}_{scale}x
```

| 输入方式 | 指纹来源 |
|---|---|
| 上传 `file_id` | 该 `file_id` 的 32 位 hex |
| 本地 `source` | 路径字符串 sha256 前 32 位 |

### 输出文件

- 目录：`VOSR_OUTPUT_DIR`（默认 `~/.vosr_server/outputs`）。
- 文件名：`{upscale_id}.mp4`。
- 入队与完成时都会打印输出路径。

### 错误格式

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "scale", "code": null}}
```

| 状态码 | 场景 |
|---|---|
| `400` | 参数非法（`file_id`/`source` 缺失或同时提供、视频不可解码） |
| `401` | 缺少/错误 API Key |
| `404` | 文件/任务不存在，或 `model` 不是已知模型 |
| `409` | 结果未就绪（`/content`），或删除正被使用的文件 |
| `422` | 目标分辨率超限、请求体校验失败 |
| `429` | 队列已满 |

## 说明

- 推理参数沿用脚本默认：one-step（`infer_steps=1`）、`dtype=bf16` DiT/DINOv2、
  DiT `tile_size=512`；`vae_tile_size` 默认 `0`（全图 VAE，更快）。可用请求字段覆盖。
- **VAE 分块（`vae_tile_size`）只为显存，与精度/时序无关**：`0`/`None` 走全图解码（更快）；
  仅在超大画面 OOM 时才设 `>0`（如 1024）。`0` 与 `None` 完全等价（详见下节）。
- **视频帧间闪烁已在根因处修复**（batch-consistent VAE 上采样），详见下节
  「[视频帧间闪烁的根因与修复](#视频帧间闪烁的根因与修复batch-consistent-vae-上采样)」。
- **图片与视频统一入口**：按 `source`/上传文件扩展名自动识别，任务对象里的 `kind` 标明
  （`image`/`video`），`object` 相应为 `image.upscale`/`video.upscale`。
- **单卡常驻**：服务在首个可见 GPU 上常驻一份模型，图片与单卡视频直接复用（无重载）；
  视频请求 `num_gpus>1` 时才改为多进程分片（每 worker 重新加载，长视频摊薄后更快）。
- **多卡**：`num_gpus>1`（或环境 `VOSR_NUM_GPUS>1`）时，按帧把视频切给 N 张卡，
  用 ffmpeg 拼接、接回音轨；取消会连同子进程一起终止（进程组 kill），不留孤儿。
- **单卡串行执行**：多个任务排队顺序处理，避免并发采样导致 CUDA OOM。
- 取消**立即生效**：`POST .../cancel` 立刻返回 `cancelled`；常驻路径在 batch 边界中断，
  多卡路径直接终止子进程组。
- **不做事前像素/显存校验**：目标过大时任务会执行到一半因 CUDA OOM 失败，错误信息直接回传
  （`error.message` 含 OOM 详情）。失败任务会自动回收显存（见「显存回收」），不影响后续任务。
- 目标过大时的处理：**分块 VAE（`vae_tile_size=1024`）可大幅降低峰值显存**——全图 VAE 的
  fp32 激活随分辨率平方增长，4K 级（约 1782万像素）在默认全图下必 OOM，设 `vae_tile_size=1024` 后峰值仅 ~8GB 即可跑通。
- 非方形输入会自动强制 DiT 分块（DiT 仅接受方形 latent）。

## 视频帧间闪烁的根因与修复（batch-consistent VAE 上采样）

**现象**：视频超分结果在不该动的区域**帧间闪烁**（静态镜头尤其明显）。

**根因**：Qwen VAE 解码器（`models/qwenimage_vae2d.py` 的 `Resample2D`）用
`nn.Upsample(scale_factor=2, mode="nearest")` 做 2× 上采样。**当输入张量非连续（non-contiguous）
时，PyTorch 的 nearest 上采样会给出 batch 不一致的结果**——同一个值在不同 batch 位置被解出
略微不同的结果。逐模块插桩可确认第一个出现差异的正是解码器最后一个 `Upsample`：
输入逐样本完全相同（spread 0），输出却逐样本不同（1440p、B=8 实测 spread ≈ 4.4）。

- 单张图**不可见**（只解 1 个样本，且输出与训练一致）。
- **视频按 `batch_size` 多帧一起解码**时，这个 batch 相关扰动让静止区域帧间抖动 → **闪烁**。
- 分块解码（`vae_tile_size>0`）每次只解小块、输入更规整，恰好没触发，所以一度看起来"分块才不闪"。

**修复**（`models/qwenimage_vae2d.py`，`Resample2D.forward`）：`upsample2d` 前强制连续：

```python
def forward(self, x):
    if self.mode == "upsample2d":
        x = x.contiguous()
    return self.resample(x)
```

**性质**：该修复是**无损**的——nearest 上采样本质是"复制"，与
`repeat_interleave(2,dim=2).repeat_interleave(2,dim=3)` **位精确相同**（实测
`torch.equal==True`、`max|diff|=0`，fp32/bf16/fp16 均如此），因此**不改变精度、与训练一致**；
成本相对该层巨大的输出可忽略，**大 batch 解码完全保留**。

> 说明：`downsample2d`（编码器用）走 `Conv2d`，**不存在**此问题，无需处理。

**实测效果**（720p→1440p，bf16，batch=8，默认全图 VAE）：

| 片段 | 修复前 帧间差 | 修复后 帧间差 |
|---|---|---|
| 静态 8 帧 | ~59 | **~0.6** |
| 运动 24 帧 | ~16.5 | **~3.3** |

修复后**全图 VAE 与分块 VAE 逐帧一致**，所以视频不再需要为了"防闪烁"而强制分块——
`vae_tile_size` 回归为纯**显存**开关（见下节）。

## 视频多 batch 时 `vae_tile_size` 怎么设

视频是**按 `batch_size` 帧一组**送进 VAE 编码/解码的，VAE 激活显存随 **batch 数 × 分辨率面积**增长。

**默认就该用 `vae_tile_size=0`（或 `None`，两者等价）——全图解码**：
- 最快（分块要做多次子解码再高斯融合）。
- 对多 batch 视频**已修复为时序稳定**（见"说明"里的 `Resample2D .contiguous()` 修复），
  不再需要为了"防闪烁"而强制分块。
- 结果与分块解码几乎逐像素一致（`0`/`None`/`1024` 之间）。

**只有以下情况才需要设 `vae_tile_size > 0`（如 `1024`）**：`batch_size` 较大 **且** 目标分辨率很高
（例如 `batch_size=8` + 1440p/4K），全图 VAE 解码出现 **CUDA OOM**。此时二选一：

| 手段 | 说明 | 代价 |
|---|---|---|
| 调小 `batch_size`（如 8→4→2） | 直接按比例降 VAE 峰值显存 | 吞吐略降 |
| 设 `vae_tile_size=1024`（或 512） | 把 VAE 解码切成瓦片，峰值显存大幅下降 | 每帧略慢（分块融合） |

推荐顺序：**先调小 `batch_size`**（改动最小、结果不变），仍不够再设 `vae_tile_size`。
两者可叠加。`vae_tile_size` **不影响画质/时序**，只是显存/速度权衡。

调用示例（8 帧一批 + 4K 爆显存时）：

```json
{"source":"/path/in.mp4","scale":2,"tile_size":512,"batch_size":8,"vae_tile_size":1024}
```

## 显存回收

- **任务处理中显存会涨到该任务的峰值**（随分辨率 × `batch_size` × 是否分块），处理完**会自动回落到基础占用**。
- **失败/取消的任务也会回收**：任务结束（无论成功、失败、取消）后服务会 `gc.collect()` + `torch.cuda.empty_cache()`，
  并把异常栈帧里的张量引用清掉，把峰值显存还给驱动。日志会打印一行
  `reclaimed <N>MB GPU cache (reserved A -> B)`。
- 因此**单卡 OOM 失败不会再残留显存**：失败后 `nvidia-smi` 会回到基础占用，后续多卡任务的 GPU0 分片不会被挤爆。
  （若发现显存仍不回落，多半是运行的是旧进程，重启服务即可。）

## 常见问题

### 提交后"服务没反应"

**提交本身是秒级返回的**（POST `/v1/upscales` 立刻返回 `202` 与任务对象），
真正的推理在后台队列里跑，**不会阻塞 HTTP 请求**。所谓"没反应"通常是：

1. **长任务：** 822 帧 720p→1440p 约 **15 分钟**（3x/4x 更久）。任务在跑，只是耗时。
   服务端终端现在会**每 ~15s 打印一次进度**（见下方"日志"），也可用 `GET .../{id}` 轮询。
2. **`curl -f` 吞掉了错误响应：** 用 `curl -sS` 而非 `-fsS`。带 `-f` 时 4xx/5xx 会返回空，
   `python -c ...["id"]` 报 `Expecting value: line 1 column 1`。
3. **`$BASE`/`$AUTH` 未设置：** `AUTH` 在**未设 `VOSR_API_KEY`** 时不要传（否则 401）；
   `BASE` 默认 `http://127.0.0.1:18303`，跨机访问要换成服务所在 IP 且确认防火墙放行。
4. **模型还在加载：** 启动后需等 `runtime[0] ready ... / Application startup complete`；
   在此之前 `/health` 的 `resident` 为 `false`。先 `curl -sS $BASE/health` 确认 `resident:true`。

排查步骤：`curl -sS $BASE/health` 看 `resident/active/queued` → 提交时 `echo "$JOB"`
确认拿到 `id` → 轮询看 `status/phase/progress` 是否在推进。

### 下载报 `curl: (23) Failure writing output to destination`

**不是服务问题，是 `-o` 指向的目录不可写。** `curl -o` 会在本地创建文件，
若输出路径所在的目录对当前用户没有写权限（例如把结果写进只读的代码仓库目录），
curl 就报 23。把结果下到可写目录即可：

```bash
curl -sSL "$BASE/v1/upscales/$UPSCALE_ID/content" -o /tmp/output_upscaled.mp4
# 或 ~/output_upscaled.mp4
```

下载前先用 `GET .../{id}` 确认 `status == "completed"`（未完成时 `/content` 返回 `409`）。

### `mat1 and mat2 must have the same dtype`

已在 `models/lightningdit.py` 的 `TimestepEmbedder.forward` 里修复（内部 fp32 与 bf16
权重对齐），无需额外处理；详见 `inference_vosr_onestep.py` 中 `run_dtype` 处的注释。

## 日志

任务运行时用 **tqdm 进度条就地刷新**（不再重复刷屏），任务结束打印一行汇总：

```
2026-10-06 18:40:17 | INFO    | vosr | job upscale_..._4420aa_2x started: kind=video scale=2.0 target=2560x1440
2026-10-06 18:40:17 | INFO    | vosr | job input=bg_video.mp4 src=1280x720 frames=822 fps=30.000 target=2560x1440 tile=512 vae_tile=None batch=8
video 2x 4420aa_2x: |██████▏   | 62%, elapsed 21s        ← 就地刷新（同一行）
2026-10-06 18:40:49 | INFO    | vosr | job upscale_..._4420aa_2x completed: output=... size=3188281B elapsed=32.3s encoder=auto
```

- 进度条只在**交互式终端**显示为动态单行；重定向到文件时按行追加（`\r`）。
- 期间的其他日志（`queued/started/completed`、`core` 的 `[Tiled Latent]` 等）会**自动避开进度条**（通过 `tqdm.write`），不会被冲掉。
- `tqdm` 不可用时自动降级为每 ~15s/≥5% 打印一行进度。
- 级别由 `VOSR_LOG_LEVEL` 控制。

## Appendix：FlashVSR 与 VOSR2 模型对比

本目录的 VOSR2 服务与同级的 FlashVSR 服务（`../FlashVSR/examples/WanVSR/infer_server.py`）是两类
定位不同的超分模型，按输入类型各自封装为常驻 OpenAI 兼容服务。以下为事实对比。

### 一句话定位

- **FlashVSR**：面向**视频**的实时流式 diffusion VSR，卖点是**时序一致性 + 实时**（~17 FPS @768×1408，A100）。
- **VOSR2（VOSR 2.0）**：面向**图像**的 one-step generative SR，卖点是**单图细节/文字保真**，vision-only，无时序模块。

### 对比表

| 维度 | FlashVSR (v1.1) | VOSR2 (VOSR 2.0) |
|---|---|---|
| 任务 | 视频超分（VSR），分辨率/内容泛化 | 图像超分（image SR） |
| 论文 | arXiv 2510.12747（清华 / OpenImagingLab） | CVPR 2026（PolyU / OPPO） |
| 生成范式 | 单步 streaming diffusion（三段蒸馏 DMD） | 单步 distill diffusion（one-step） |
| 主干 | Wan2.1 系 DiT（`wan_video_dit.py`，5.67GB ckpt） | 自研 1.4B DiT：`dim=1536, depth=36, heads=24, rope/rmsnorm/swiglu` |
| 条件编码 | 文本 prompt + `LQ_proj_in`（时序 LQ 投影，575MB） | **vision-only**：DINOv2-L 特征（1024-d）+ LQ，无文本编码器 |
| 核心加速 | **Locality-Constrained Sparse Attention (LCSA)** + **Tiny Conditional Decoder** | tiled DiT（`tile_size=512`）+ O(N) 的 Qwen-Image VAE |
| VAE/Decoder | Wan2.1 VAE（507MB）+ `TCDecoder`（189MB） | Qwen-Image VAE **2D** 变体（去掉 3D 时间开销） |
| 时序建模 | **有**（cross-KV 流式缓存、`8n+1` 帧 chunk、`kv_ratio/local_range`） | **无**（逐帧独立；视频按帧序列处理） |
| 帧数约束 | 需 `8n+1` 帧（流式最小 ~25 帧 / 1 chunk） | 单帧即输入，无时序约束 |
| 输出 | 视频 MP4（保留音轨 / FPS） | 图像 PNG |
| 分辨率/倍率 | 推荐 **4×**（代码经验值，过高易出内存/伪影） | 训练分辨率 512，倍率 **1–8×** |
| 显存控制 | `topk_ratio`（sparse）、`MAX_TARGET_PIXELS` | DiT/VAE 双 tiling（`tile_size` / `vae_tile_size`） |
| 硬件 | Block-Sparse-Attn 在 **A100/A800** 加速最佳，H200 有限；H800/RTX 未验证 | 常规 attention，无特殊 kernel 依赖 |
| 服务输入类型 | 视频 + 图片目录（多帧序列） | 图片 + 视频（逐帧，`kind` 自动识别） |
| 开源 | 代码 + v1/v1.1 权重；数据集 VSR-120K "coming soon" | 代码 + 权重 + ScreenSR 基准 + 训练/蒸馏代码 |

### 关键差异（工程含义）

1. **时序一致性**：FlashVSR 有 cross-KV 流式建模，连续帧稳定；VOSR2 无时序模块，逐帧独立 →
   视频逐帧跑**可能有闪烁**，但单帧质量 / 文字更好。
2. **速度与场景**：FlashVSR 为视频实时优化（流式 chunk + 稀疏注意力）；VOSR2 为单图质量优化
   （one-step、tiled、大 VAE）。
3. **条件方式**：FlashVSR 走 Wan2.1 文本 encoder（服务里 prompt 实际置空）；VOSR2 纯视觉 DINOv2
   条件，省掉文本编码。
4. **倍率**：FlashVSR 官方强烈建议 4×（其服务默认 2×，`scale∈[1,4]`）；VOSR2 支持 1–8×。
5. **部署依赖**：FlashVSR 依赖 `Block-Sparse-Attention` 编译与特定 GPU；VOSR2 无稀疏 kernel 依赖，更易部署。

### 选型建议

- **要视频、要时序稳定 / 实时** → FlashVSR（4×，A100/A800）。
- **要单图、要文字 / 细节保真、需要 1–8× 或大图 tiling** → VOSR2。
- **两者结合**：两个常驻服务分别承接视频与图像，按输入类型路由即可。

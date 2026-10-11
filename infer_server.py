#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常驻 GPU 的 VOSR2 one-step 视频超分服务（OpenAI 兼容 API）。

基于 `inference_vosr_onestep_pipeline.py` 改造：
  - 启动时加载一次 VOSR2 模型并常驻显存，重复任务不重新加载权重；
  - OpenAI 兼容风格 API：/v1/files 上传、/v1/upscales 提交/查询/取消/下载；
  - Bearer Token 鉴权（设置 VOSR_API_KEY 后启用）；
  - 异步排队、单卡串行推理；输出用 ffmpeg 编码并保留原视频音轨；
  - 多卡：GPU0 的 shard 用常驻模型在进程内推理，GPU1..N-1 用常驻 worker 池
    （模型只加载一次，num_gpus 不变时跨请求复用），详见 vosr_multigpu.py；
  - 逐帧独立超分（VOSR2 无时序模块），输出帧数 == 输入帧数。

推理参数沿用脚本默认：one-step（infer_steps=1）、bf16 DiT、tile 分块。
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional

import cv2
import numpy as np

# Reduces PyTorch allocator fragmentation, which otherwise causes "out of memory"
# even when enough total memory is free (the allocator can't find a contiguous
# block). Inherited by the multi-GPU subprocess workers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional; falls back to periodic log lines
    tqdm = None



# --------------------------------------------------------------------------
# 路径：以仓库根为工作目录，保证相对 preset/... 路径可解析
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # noqa: PTH
sys.path.insert(0, str(BASE_DIR))

import vosr_core as core                        # noqa: E402

# dynamo/torch.compile must never hard-fail when driven from worker threads;
# degrade to eager instead (quality unchanged).
try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    pass

# Web 依赖（服务必需）。放在鉴权依赖定义之前，Header 需要在 def 时可用。
try:
    import uvicorn
    from fastapi import Depends, FastAPI, File, Header, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as _exc:  # pragma: no cover
    raise SystemExit(
        "infer_server.py 需要 Web 依赖，请先安装：\n"
        "    pip install fastapi uvicorn python-multipart\n"
        f"(import error: {_exc!r})") from _exc

# --------------------------------------------------------------------------
# 配置：命令行参数优先，其次 VOSR_* 环境变量，最后内置默认值
# --------------------------------------------------------------------------
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_arg_parser() -> argparse.ArgumentParser:
    """Every option's default comes from its VOSR_* env var, so a CLI flag
    overrides the env var (which overrides the built-in default)."""
    p = argparse.ArgumentParser(
        prog="infer_server.py",
        description="VOSR2 resident upscale server (OpenAI-compatible API). "
                    "CLI args override VOSR_* environment variables.")
    p.add_argument("--host", default=_env("VOSR_HOST", "0.0.0.0"),
                   help="listen address (env VOSR_HOST)")
    p.add_argument("--port", type=int, default=int(_env("VOSR_PORT", "18303")),
                   help="listen port (env VOSR_PORT)")
    p.add_argument("--api-key", default=_env("VOSR_API_KEY", ""),
                   help="Bearer token; empty disables auth (env VOSR_API_KEY)")
    p.add_argument("--ckpt", default=_env("VOSR_CKPT", str(BASE_DIR / "preset" / "ckpts" / "VOSR2")),
                   help="VOSR2 model dir (env VOSR_CKPT)")
    p.add_argument("--work-dir", default=_env("VOSR_WORK_DIR", str(Path.home() / ".vosr_server")),
                   help="inputs/jobs work dir (env VOSR_WORK_DIR)")
    p.add_argument("--output-dir", default=_env("VOSR_OUTPUT_DIR", ""),
                   help="result dir (default <work-dir>/outputs; env VOSR_OUTPUT_DIR)")
    p.add_argument("--encoder", choices=["auto", "libx264", "nvenc"], default=_env("VOSR_ENCODER", "auto"),
                   help="video encoder (env VOSR_ENCODER)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default=_env("VOSR_DTYPE", "bf16"),
                   help="DiT/DINOv2 compute dtype; VAE stays fp32 (env VOSR_DTYPE)")
    p.add_argument("--tile-size", type=int, default=int(_env("VOSR_TILE_SIZE", "512")),
                   help="default DiT tile size in px (env VOSR_TILE_SIZE)")
    p.add_argument("--vae-tile-size", type=int, default=int(_env("VOSR_VAE_TILE_SIZE", "0")),
                   help="default VAE tile size; 0=full-image (env VOSR_VAE_TILE_SIZE)")
    p.add_argument("--batch-size", type=int, default=int(_env("VOSR_BATCH_SIZE", "8")),
                   help="default frames per forward (env VOSR_BATCH_SIZE)")
    p.add_argument("--seed", type=int, default=int(_env("VOSR_SEED", "42")),
                   help="default random seed; request field `seed` overrides (env VOSR_SEED)")
    p.add_argument("--num-gpus", type=int, default=int(_env("VOSR_NUM_GPUS", "1")),
                   help="default frame-level sharding for video; 1=resident single GPU (env VOSR_NUM_GPUS)")
    p.add_argument("--max-queue", type=int, default=int(_env("VOSR_MAX_QUEUE", "8")),
                   help="max queued jobs (env VOSR_MAX_QUEUE)")
    p.add_argument("--model-name", default=_env("VOSR_MODEL_NAME", "VOSR2-1.4B-OneStep"),
                   help="reported model name (env VOSR_MODEL_NAME)")
    p.add_argument("--log-level", default=_env("VOSR_LOG_LEVEL", "INFO").upper(),
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="logging level (env VOSR_LOG_LEVEL)")
    return p


# 运行期配置（由 apply_config 填充；默认来自环境变量）。类型标注让 Pylance
# 知道它们在 apply_config 后的实际类型，而不是从 None 初始值推断出的 None。
HOST: str = ""
PORT: int = 0
API_KEY: str = ""
ENCODER: str = "auto"
DTYPE: str = "bf16"
MODEL_NAME: str = ""
LOG_LEVEL: str = "INFO"
CKPT: Path = Path()
WORK_DIR: Path = Path()
OUTPUT_DIR: Path = Path()
INPUT_DIR: Path = Path()
JOB_DIR: Path = Path()
FILE_META_DIR: Path = Path()
DEFAULT_TILE: int = 0
DEFAULT_VAE_TILE: int = 0
DEFAULT_BATCH: int = 1
DEFAULT_SEED: int = 42
NUM_GPUS: int = 1
MAX_QUEUE: int = 8
MODEL_CREATED = int(time.time())

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
UPSCALE_ID_RE = re.compile(r"^upscale_[0-9a-f]{32}(_[0-9a-f]{6}_[0-9.]+x)?$")
FILE_ID_RE = re.compile(r"^file_[0-9a-f]{32}$")
TERMINAL = {"completed", "failed", "cancelled"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}

# --------------------------------------------------------------------------
# 日志（+ 进度条）
# --------------------------------------------------------------------------
# 当前正在跑的任务的 tqdm 进度条（服务串行，同一时刻最多一个）。
_PROGRESS: dict[str, Any] = {"bar": None}


class _TqdmStreamHandler(logging.StreamHandler):
    """StreamHandler that cooperates with an active tqdm progress bar: log
    lines are written through ``bar.write`` so they don't corrupt the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            bar = _PROGRESS.get("bar")
            if bar is not None:
                # Write to the handler's stream (stderr) rather than tqdm's
                # stdout default, so lines aren't lost in stdout's buffer when
                # logs are redirected to a file.
                bar.write(msg, file=self.stream)
            else:
                self.stream.write(msg + "\n")
                self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger("vosr")
if not logger.handlers:
    _handler = _TqdmStreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.propagate = False

# vosr_core (imported above) called logging.basicConfig, which attached a plain
# StreamHandler to the root logger (format "[time] LEVEL msg"). Replace it with a
# tqdm-aware one so core's messages (e.g. "[Tiled Latent]") also play nice.
_root = logging.getLogger()
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_root_handler = _TqdmStreamHandler()
_root_handler.setFormatter(logging.Formatter(
    "[\033[34m%(asctime)s\033[0m] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root.addHandler(_root_handler)
if _root.level in (logging.NOTSET, 0):
    _root.setLevel(logging.INFO)


def _job_label(job: dict[str, Any]) -> str:
    jid = str(job.get("id", ""))
    short = "_".join(jid.split("_")[-2:]) if "_" in jid else jid
    req = job.get("request") or {}
    label = str(req.get("kind", job.get("kind", "?")))
    scale = req.get("scale")
    if isinstance(scale, (int, float)):
        label += f" {float(scale):g}x"
    return f"{label} {short}"


def _open_bar(job: dict[str, Any]):
    if tqdm is None:
        return None

    class _JobTqdm(tqdm):
        """tqdm that renders its postfix without the automatic ", " prefix, so
        the line ends with e.g. "62%, elapsed 21s" instead of ", 62%, elapsed 21s"."""

        def __str__(self) -> str:
            out = super().__str__()
            p = self.postfix
            if p:
                pre = ", " + p
                if out.endswith(pre):
                    out = out[: -len(pre)] + " " + p
            return out

    try:
        # leave=True: keep the final (100%) bar line in the output/log instead of
        # erasing it on close.
        bar = _JobTqdm(total=100.0, desc=_job_label(job), dynamic_ncols=True, leave=True,
                       bar_format="{desc}: |{bar}|{postfix}", mininterval=0.5)
    except Exception:
        return None
    _PROGRESS["bar"] = bar
    return bar


def _close_bar() -> None:
    bar = _PROGRESS.get("bar")
    if bar is not None:
        try:
            bar.close()
        except Exception:
            pass
        _PROGRESS["bar"] = None


def apply_config(a: argparse.Namespace) -> None:
    """Assign resolved config to module globals, (re)create dirs, set log level."""
    global HOST, PORT, API_KEY, CKPT, WORK_DIR, OUTPUT_DIR, ENCODER, DTYPE
    global DEFAULT_TILE, DEFAULT_VAE_TILE, DEFAULT_BATCH, DEFAULT_SEED, NUM_GPUS, MAX_QUEUE
    global MODEL_NAME, LOG_LEVEL, INPUT_DIR, JOB_DIR, FILE_META_DIR

    HOST = str(a.host)
    PORT = int(a.port)
    API_KEY = str(a.api_key or "")
    CKPT = Path(a.ckpt)
    WORK_DIR = Path(a.work_dir)
    OUTPUT_DIR = Path(a.output_dir) if a.output_dir else WORK_DIR / "outputs"
    ENCODER = str(a.encoder)
    DTYPE = str(a.dtype)
    DEFAULT_TILE = int(a.tile_size)
    DEFAULT_VAE_TILE = int(a.vae_tile_size)
    DEFAULT_BATCH = int(a.batch_size)
    DEFAULT_SEED = int(a.seed)
    NUM_GPUS = int(a.num_gpus)
    MAX_QUEUE = int(a.max_queue)
    MODEL_NAME = str(a.model_name)
    LOG_LEVEL = str(a.log_level).upper()
    logger.setLevel(LOG_LEVEL)

    INPUT_DIR = WORK_DIR / "inputs"
    JOB_DIR = WORK_DIR / "jobs"
    FILE_META_DIR = WORK_DIR / "file_meta"
    for directory in (INPUT_DIR, OUTPUT_DIR, JOB_DIR, FILE_META_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# 模块导入时先用「环境变量默认值」初始化（CLI 在 __main__ 里再覆盖）
apply_config(build_arg_parser().parse_args([]))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _to_unix(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(str(ts)).timestamp())
    except (ValueError, TypeError):
        return None


def _epoch(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return None


def build_upscale_id(file_id: Optional[str], source: str, scale: float) -> str:
    if file_id is not None and FILE_ID_RE.fullmatch(file_id):
        prefix = file_id[len("file_"):]
    else:
        prefix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:32]
    return f"upscale_{prefix}_{uuid.uuid4().hex[:6]}_{scale:g}x"


# --------------------------------------------------------------------------
# OpenAI 兼容错误 + 鉴权
# --------------------------------------------------------------------------
class OpenAIHTTPException(Exception):
    def __init__(self, status_code: int, message: str, type: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None) -> None:
        self.status_code = status_code
        self.body = {"message": message, "type": type, "param": param, "code": code}


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not authorization:
        raise OpenAIHTTPException(401, "Missing Authorization header", "authentication_error", code="invalid_api_key")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer" or not hmac.compare_digest(parts[1], API_KEY):
        raise OpenAIHTTPException(401, "Invalid API key", "authentication_error", code="invalid_api_key")


# --------------------------------------------------------------------------
# ckpt args -> SimpleNamespace / 视频信息 / ffmpeg
# --------------------------------------------------------------------------
def load_ckpt_args() -> dict[str, Any]:
    path = CKPT / "args.json"
    if not path.is_file():
        raise RuntimeError(f"args.json not found under {CKPT}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def probe_video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    info = {
        "kind": "video",
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    if info["width"] <= 0 or info["height"] <= 0:
        raise ValueError(f"cannot read video dimensions: {path}")
    if info["fps"] <= 0:
        info["fps"] = 25.0
    return info


def probe_input(path: Path) -> dict[str, Any]:
    """Detect image vs video by extension and return kind + dimensions."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return {"kind": "image", "width": int(w), "height": int(h), "fps": 0.0, "frames": 1}
    if ext in VIDEO_EXTS:
        return probe_video_info(path)
    raise ValueError(f"unsupported input type: {path.suffix or '(no extension)'}")


def _visible_gpus() -> list[int]:
    devs = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if devs:
        return list(range(len(devs)))  # indices are remapped by CUDA itself
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return list(range(n))


def _has_audio(path: Path) -> bool:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return bool(completed.stdout.strip())


def remux_audio(video_only: Path, source: Path, output: Path) -> None:
    source_has_audio = _has_audio(source)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(source),
         "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)],
        check=True,
    )
    if source_has_audio and not _has_audio(output):
        logger.warning("audio lost during remux: source=%s output=%s", source.name, output.name)
    elif source_has_audio:
        logger.info("audio preserved from source (%s)", source.name)
    else:
        logger.info("source has no audio track (%s)", source.name)


def _encoder_args(encoder: str) -> list[str]:
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
                "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]


class _VideoWriter:
    """Streams BGR uint8 frames to ffmpeg (rawvideo pipe). Encoder fallback handled by caller."""

    def __init__(self, path: Path, width: int, height: int, fps: float, encoder: str) -> None:
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s:v", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-an"]
        cmd += _encoder_args(encoder)
        cmd += ["-movflags", "+faststart", str(path)]
        self.path = path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, bgr: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def close(self) -> int:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        return self.proc.wait()

    def abort(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()


def release_host_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def reclaim_gpu_memory() -> None:
    """Return cached (reserved-but-unused) GPU memory to the driver.

    Called after every task (success, failure, cancel). For a failed task the
    caller has already run ``traceback.clear_frames`` so the traceback no longer
    keeps the failing frame's tensors alive; only then can ``empty_cache`` hand
    the peak allocation back to the driver. Without this, an OOM'd task leaves
    its ~peak (tens GB) reserved in the process until the next task reclaims it.
    """
    release_host_memory()
    try:
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved() / 2**20
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved() / 2**20
            if before - after >= 1.0:
                logger.info("reclaimed %.0fMB GPU cache (reserved %.0fMB -> %.0fMB)",
                            before - after, before, after)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 取消
# --------------------------------------------------------------------------
class InferenceCancelled(Exception):
    """任务被取消时中断推理/编码。"""


# --------------------------------------------------------------------------
# 常驻运行时（线程内执行单个超分任务）
# --------------------------------------------------------------------------
class VOSRRuntime:
    def __init__(self, device_index: int = 0) -> None:
        loaded_at = time.monotonic()
        self.device_index = int(device_index)
        self.base = load_ckpt_args()
        self.device = (f"cuda:{self.device_index}"
                       if torch.cuda.is_available() else "cpu")
        self.run_args = self._make_args(
            tile_size=DEFAULT_TILE if DEFAULT_TILE > 0 else 0,
            vae_tile_size=(int(DEFAULT_VAE_TILE) or None),
        )
        self.run_args.dtype = DTYPE
        self.run_args.checkpoint = str(CKPT)
        name = torch.cuda.get_device_name(self.device_index) if torch.cuda.is_available() else "cpu"
        logger.info("GPU[%d]: %s CUDA=%s dtype=%s", self.device_index, name,
                    getattr(torch.version, "cuda", None), DTYPE)
        self.vae: Any = None
        self.venc: Any = None
        self.model: Any = None
        self.vosr: Any = None
        self.light_decoder: Any = None
        self.sample_fn: Any = None
        self._load_models()
        self.jobs_completed = 0
        self.load_seconds = round(time.monotonic() - loaded_at, 3)
        logger.info("runtime[%d] ready: load_seconds=%.3f", self.device_index, self.load_seconds)

    def _load_models(self) -> None:
        self.vae, self.venc, self.model, self.vosr, self.light_decoder = core.load_models(
            self.run_args, self.device)
        self.sample_fn = self.vosr.sample_onestep

    def ensure_loaded(self) -> None:
        # The resident model is kept loaded for the server's lifetime (single-GPU
        # and image requests use it; multi-GPU uses it for GPU0's shard in-process),
        # so this is normally a no-op.
        if self.model is None:
            logger.info("loading resident model on GPU %d ...", self.device_index)
            self._load_models()

    def _make_args(self, tile_size: int, vae_tile_size: Optional[int]):
        a = SimpleNamespace(**self.base)
        a.ae_type = getattr(a, "ae_type", "qwen")
        if getattr(a, "auxiliary_time_cond", None) is None:
            a.auxiliary_time_cond = True
        a.auxiliary_time_cond = bool(a.auxiliary_time_cond)
        a.tile_size = int(tile_size)
        a.tile_overlap = 32
        a.vae_tile_size = vae_tile_size
        a.vae_tile_overlap = None
        a.posterior_mode = True
        a.infer_steps = 1
        a.align_method = "wavelet"
        a.dtype = DTYPE
        a.checkpoint = str(CKPT)
        return a

    def _job_params(self, scale, tile_size, vae_tile_size, batch_size, align_method, src_w, src_h):
        ts = int(tile_size) if tile_size is not None else (DEFAULT_TILE if DEFAULT_TILE > 0 else 0)
        if ts <= 0 and src_w != src_h:
            ts = 512  # DiT needs square latents; force tiling on non-square inputs
        vts = vae_tile_size if vae_tile_size is not None else (int(DEFAULT_VAE_TILE) or None)
        if vts is not None and vts <= 0:
            vts = None
        bs = max(1, int(batch_size) if batch_size is not None else DEFAULT_BATCH)
        job_args = self._make_args(tile_size=ts, vae_tile_size=vts)
        job_args.align_method = align_method
        job_args.upscale = scale
        return job_args, ts, vts, bs

    @torch.no_grad()
    def upscale_image(self, source: Path, output: Path, scale: float, seed: int = 42,
                      tile_size: Optional[int] = None, vae_tile_size: Optional[int] = None,
                      align_method: str = "wavelet",
                      progress_cb: Optional[Callable[[float, str], None]] = None,
                      cancel_check: Optional[Callable[[], bool]] = None) -> dict:
        """Single-image super-resolution -> PNG."""
        from PIL import Image
        from torchvision import transforms
        self.ensure_loaded()
        source, output = source.resolve(), output.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if not 1.0 <= scale <= 8.0:
            raise ValueError("scale must be between 1 and 8")
        if progress_cb is not None:
            progress_cb(5.0, "preparing")
        started = time.monotonic()

        raw = Image.open(source).convert("RGB")
        src_w, src_h = raw.size
        target_w, target_h = int(round(src_w * scale)), int(round(src_h * scale))
        job_args, ts, vts, _bs = self._job_params(scale, tile_size, vae_tile_size, None, align_method, src_w, src_h)
        logger.info("image input=%s src=%dx%d target=%dx%d tile=%s vae_tile=%s",
                    source.name, src_w, src_h, target_w, target_h, ts, vts)
        if cancel_check is not None and cancel_check():
            raise InferenceCancelled("cancelled before start")

        input_img = raw.resize((target_w, target_h), Image.Resampling.BICUBIC)
        lq = transforms.ToTensor()(input_img).unsqueeze(0).to(self.device) * 2.0 - 1.0
        torch.manual_seed(int(seed))
        if job_args.tile_size > 0:
            sr = core.tiled_latent_inference(self.model, self.vae, self.venc, lq, job_args,
                                            device=self.device, light_decoder=self.light_decoder)
        else:
            lq_latent, m, s = core.encode_dispatch(self.vae, lq, job_args, self.device)
            z_fea = core.get_venc_features(self.venc, lq, job_args)
            mdt = next(self.model.parameters()).dtype
            lq_latent = lq_latent.to(mdt)
            z_fea = [f.to(mdt) for f in z_fea]
            sr_latent = self.sample_fn(self.model, lq_latent, n_steps=1, venc_fea=z_fea)
            sr = core.decode_dispatch(self.vae, sr_latent, job_args, m, s, self.light_decoder)
        if progress_cb is not None:
            progress_cb(90.0, "encoding")

        sr_img = transforms.ToPILImage()(sr[0].cpu() * 0.5 + 0.5)
        if job_args.align_method == "adain":
            sr_img = core.adain_color_fix(sr_img, input_img)
        elif job_args.align_method == "wavelet":
            sr_img = core.wavelet_color_fix(sr_img, input_img)
        output.parent.mkdir(parents=True, exist_ok=True)
        sr_img.save(output)

        torch.cuda.empty_cache()
        release_host_memory()
        if progress_cb is not None:
            progress_cb(100.0, "completed")
        self.jobs_completed += 1
        return {"status": "completed", "kind": "image", "scale": scale,
                "width": target_w, "height": target_h, "frames": 1, "fps": 0.0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "resident": True, "engine_load_seconds": self.load_seconds}

    def process(
        self,
        source: Path,
        output: Path,
        scale: float,
        seed: int = 42,
        tile_size: Optional[int] = None,
        vae_tile_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        align_method: str = "wavelet",
        encoder: str = "auto",
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        self.ensure_loaded()
        source = source.resolve()
        output = output.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if not 1.0 <= scale <= 8.0:
            raise ValueError("scale must be between 1 and 8")

        def report(p: float, phase: str) -> None:
            if progress_cb is not None:
                progress_cb(p, phase)

        def cancelled() -> bool:
            return cancel_check is not None and cancel_check()

        info = probe_video_info(source)
        src_w, src_h, fps = info["width"], info["height"], info["fps"]
        total = info["frames"]
        target_w = int(round(src_w * scale))
        target_h = int(round(src_h * scale))

        job_args, ts, vts, bs = self._job_params(
            scale, tile_size, vae_tile_size, batch_size, align_method, src_w, src_h)

        output.parent.mkdir(parents=True, exist_ok=True)
        video_only = output.with_suffix(".video-only.mp4")
        total_started = time.monotonic()
        timings: dict[str, float] = {}

        report(1.0, "preparing")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {source}")
        writer: Optional[_VideoWriter] = None
        frames_done = 0
        pending: list[np.ndarray] = []
        pending_seeds: list[int] = []

        def flush() -> None:
            nonlocal writer, frames_done
            if not pending:
                return
            if cancelled():
                raise InferenceCancelled("inference cancelled")
            outs = core.upscale_frames_batch(
                pending, self.model, self.vae, self.venc, self.sample_fn,
                job_args, self.device, self.light_decoder, list(pending_seeds),
            )
            for out_bgr, _pil in outs:
                if writer is None:
                    h, w = out_bgr.shape[:2]
                    enc = encoder
                    writer = _VideoWriter(video_only, w, h, fps, enc)
                writer.write(out_bgr)
                frames_done += 1
            pending.clear()
            pending_seeds.clear()
            if total > 0:
                report(5.0 + 85.0 * frames_done / total, "upscaling")

        try:
            report(3.0, "upscaling")
            stage = time.monotonic()
            frame_idx = 0
            while True:
                if cancelled():
                    raise InferenceCancelled("inference cancelled")
                ok, frame = cap.read()
                if not ok:
                    break
                pending.append(frame)
                pending_seeds.append(int(seed) + frame_idx)  # per-frame distinct noise
                frame_idx += 1
                if len(pending) >= bs:
                    flush()
            flush()
            timings["inference"] = round(time.monotonic() - stage, 3)

            if writer is None:
                raise RuntimeError("no frames were produced")
            stage = time.monotonic()
            rc = writer.close()
            writer = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg encoder exited with code {rc}")
            timings["encode"] = round(time.monotonic() - stage, 3)

            report(97.0, "audio_remux")
            stage = time.monotonic()
            remux_audio(video_only, source, output)
            timings["audio_remux"] = round(time.monotonic() - stage, 3)

            report(100.0, "completed")
            elapsed = round(time.monotonic() - total_started, 3)
            self.jobs_completed += 1
            return {
                "status": "completed", "scale": scale,
                "width": target_w, "height": target_h,
                "frames": frames_done, "fps": round(fps, 3),
                "elapsed_seconds": elapsed, "encoder": encoder,
                "resident": True, "engine_load_seconds": self.load_seconds,
                "timings": timings,
            }
        finally:
            if writer is not None:
                writer.abort()
            cap.release()
            video_only.unlink(missing_ok=True)
            pending.clear()
            torch.cuda.empty_cache()
            release_host_memory()


# --------------------------------------------------------------------------
# 多卡帧级数据并行：把视频的连续帧区间分给多张卡并行处理，再拼接
# --------------------------------------------------------------------------
def _process_multigpu(ngpu: int, runtime: "VOSRRuntime", pool, source: Path, output: Path,
                      scale: float, seed: int, tile_size, vae_tile_size, batch_size,
                      align_method: str, encoder: str,
                      progress_cb=None, cancel_check=None) -> dict:
    """帧级多卡：GPU0 的 shard 用常驻模型在进程内推理（常驻模型不卸载），
    GPU1..N-1 用常驻 worker 池（模型只加载一次），最后拼接 + 接回音轨。"""
    import vosr_multigpu as mg

    source, output = source.resolve(), output.resolve()
    _job_args, ts, vts, bs = runtime._job_params(
        scale, tile_size, vae_tile_size, batch_size, align_method, 1, 1)
    cfg = mg.build_config(
        checkpoint=str(CKPT), source=str(source), output=str(output),
        scale=scale, seed=seed, tile_size=ts, vae_tile_size=vts, batch_size=bs,
        align_method=align_method, dtype=DTYPE, num_gpus=ngpu,
    )
    physical = core._visible_gpus()
    n = max(1, min(ngpu, len(physical)))

    def _pct(p, phase=None):
        if progress_cb is not None:
            progress_cb(5.0 + 0.9 * float(p), "upscaling")

    # GPU0 shard in-process on the resident model (never released); GPU1.. on the pool.
    resident_models = (runtime.vae, runtime.venc, runtime.model, runtime.vosr, runtime.light_decoder)
    executors = [mg.inproc_executor(resident_models, runtime.device)]
    if n > 1:
        pool.ensure(n - 1)
        for g in range(1, n):
            executors.append(mg.pool_executor(pool, physical[g]))

    started = time.monotonic()
    try:
        mg.run_multigpu(cfg, executors, progress_cb=_pct, cancel_check=cancel_check,
                        log=logger.info, devices=physical[:n])
    except Exception as exc:
        if mg.is_cancelled(exc):
            raise InferenceCancelled("inference cancelled")
        raise
    if progress_cb is not None:
        progress_cb(100.0, "completed")
    info = probe_video_info(output)
    runtime.jobs_completed += 1
    return {
        "status": "completed", "kind": "video", "scale": scale,
        "width": info["width"], "height": info["height"], "frames": info["frames"],
        "fps": round(info["fps"], 3), "elapsed_seconds": round(time.monotonic() - started, 3),
        "num_gpus": ngpu, "resident": True,
        "engine_load_seconds": runtime.load_seconds,
    }


# --------------------------------------------------------------------------
# 请求模型 + Worker（异步排队、串行执行）
# --------------------------------------------------------------------------
class Worker:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=MAX_QUEUE)
        self.runtimes: list[VOSRRuntime] = []
        self.runner_task: Optional["asyncio.Task"] = None
        self.pool: Any = None
        self.load_jobs()

    @property
    def runtime(self) -> Optional[VOSRRuntime]:
        return self.runtimes[0] if self.runtimes else None

    def load_jobs(self) -> None:
        for path in JOB_DIR.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job_id = str(job.get("id", ""))
            if not UPSCALE_ID_RE.fullmatch(job_id):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "failed"
                job["phase"] = "failed"
                job["finished_at"] = now_iso()
                job["error"] = {"message": "server restarted before the job completed",
                                "type": "server_error", "param": None, "code": None}
                atomic_json(path, job)
            self.jobs[job_id] = job

    def persist(self, job: dict[str, Any]) -> None:
        atomic_json(JOB_DIR / f"{job['id']}.json", job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": job.get("id"),
            "object": job.get("object", "video.upscale"),
            "model": job.get("model", MODEL_NAME),
            "kind": job.get("kind"),
            "file_id": job.get("file_id"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "progress": job.get("progress", 0.0),
            "created_at": _to_unix(job.get("created_at")),
            "started_at": _to_unix(job.get("started_at")),
            "finished_at": _to_unix(job.get("finished_at")),
            "queue_seconds": self._queue_seconds(job),
            "elapsed_seconds": self._elapsed(job),
            "total_seconds": self._total_seconds(job),
            "request": dict(job.get("request") or {}),
            "result": dict(job["result"]) if isinstance(job.get("result"), dict) else None,
            "error": job.get("error"),
        }
        if payload["status"] == "queued":
            # asyncio.Queue has no public accessor for pending items; _queue is stable.
            queued = [jid for jid in list(self.queue._queue)  # pyright: ignore[reportAttributeAccessIssue]
                      if self.jobs.get(jid, {}).get("status") == "queued"]
            try:
                payload["queue_position"] = queued.index(job["id"]) + 1
            except ValueError:
                payload["queue_position"] = 1
        return payload

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        request = job.get("request") or {}
        error = job.get("error")
        return {
            "id": job.get("id"),
            "object": job.get("object", "video.upscale"),
            "model": job.get("model", MODEL_NAME),
            "kind": job.get("kind"),
            "file_id": job.get("file_id"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "progress": job.get("progress", 0.0),
            "scale": request.get("scale"),
            "source_width": request.get("source_width"),
            "source_height": request.get("source_height"),
            "source_frames": request.get("source_frames"),
            "source_fps": request.get("source_fps"),
            "source_duration_seconds": request.get("source_duration_seconds"),
            "target_width": request.get("target_width"),
            "target_height": request.get("target_height"),
            "created_at": _to_unix(job.get("created_at")),
            "started_at": _to_unix(job.get("started_at")),
            "finished_at": _to_unix(job.get("finished_at")),
            "queue_seconds": self._queue_seconds(job),
            "elapsed_seconds": self._elapsed(job),
            "total_seconds": self._total_seconds(job),
            "error": error.get("message") if isinstance(error, dict) else error,
        }

    def summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        ordered = sorted(self.jobs.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True)
        return [self.summary(job) for _, job in ordered[:limit]]

    @staticmethod
    def _elapsed(job: dict[str, Any]) -> Optional[float]:
        result = job.get("result")
        if isinstance(result, dict) and result.get("elapsed_seconds") is not None:
            return round(float(result["elapsed_seconds"]), 1)
        started = _epoch(job.get("started_at"))
        if started is None:
            return None
        finished = _epoch(job.get("finished_at"))
        end = finished if finished is not None else time.time()
        return round(max(0.0, end - started), 1)

    @staticmethod
    def _queue_seconds(job: dict[str, Any]) -> Optional[float]:
        created = _epoch(job.get("created_at"))
        started = _epoch(job.get("started_at"))
        if created is None or started is None:
            return None
        return round(max(0.0, started - created), 1)

    @staticmethod
    def _total_seconds(job: dict[str, Any]) -> Optional[float]:
        created = _epoch(job.get("created_at"))
        if created is None:
            return None
        finished = _epoch(job.get("finished_at"))
        end = finished if finished is not None else time.time()
        return round(max(0.0, end - created), 1)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        gpus = _visible_gpus()
        primary = gpus[0] if gpus else 0
        logger.info("loading resident runtime on GPU index %d (multi-GPU videos shard to GPUs 1..)",
                    primary)
        self.runtimes = await loop.run_in_executor(None, lambda: [VOSRRuntime(primary)])
        # Persistent per-GPU workers for multi-GPU video (GPU1..); loaded once and
        # reused while num_gpus is unchanged. GPU0's shard runs in-process on the
        # resident model, so it stays loaded and is never released.
        import vosr_multigpu as mg

        def _base_cfg() -> dict:
            return mg.build_config(checkpoint=str(CKPT), dtype=DTYPE, codec="mp4v",
                                   num_gpus=1, source="/dev/null", output="/tmp/_vosr_pool")
        self.pool = mg.WorkerPool(_base_cfg(), core._visible_gpus()[1:], log=logger.info)
        self.runner_task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        if self.runner_task:
            self.runner_task.cancel()
            try:
                await self.runner_task
            except asyncio.CancelledError:
                pass
        if getattr(self, "pool", None) is not None:
            await asyncio.to_thread(self.pool.close)

    async def submit(self, request: "UpscaleRequest") -> dict[str, Any]:
        if request.model is not None and request.model != MODEL_NAME:
            raise OpenAIHTTPException(404, f"model '{request.model}' not found",
                                      type="not_found_error", param="model", code="model_not_found")
        if (request.source is None) == (request.file_id is None):
            raise OpenAIHTTPException(400, "provide exactly one of 'file_id' or 'source'", param="file_id")

        if request.file_id is not None:
            if not FILE_ID_RE.fullmatch(request.file_id):
                raise OpenAIHTTPException(400, "invalid file_id", param="file_id")
            matches = list(INPUT_DIR.glob(f"{request.file_id}.*"))
            if not matches:
                raise OpenAIHTTPException(404, "uploaded file not found", type="not_found_error",
                                          param="file_id", code="file_not_found")
            source = matches[0]
            source_provider = "upload"
        else:
            assert request.source is not None  # guaranteed by the XOR check above
            source = Path(request.source)
            if not source.is_file():
                raise OpenAIHTTPException(404, "source path not found", type="not_found_error",
                                          param="source", code="source_not_found")
            source_provider = "path"

        try:
            info = await asyncio.to_thread(probe_input, source)
        except (ValueError, OSError) as exc:
            raise OpenAIHTTPException(400, f"cannot read input: {exc}", param="file_id")

        kind = info["kind"]
        target_width = int(round(info["width"] * request.scale))
        target_height = int(round(info["height"] * request.scale))

        active_count = sum(1 for item in self.jobs.values() if item.get("status") not in TERMINAL)
        if active_count >= MAX_QUEUE:
            raise OpenAIHTTPException(429, "queue is full", type="rate_limit_exceeded", code="queue_full")

        job_id = build_upscale_id(request.file_id, str(source), request.scale)
        while job_id in self.jobs:
            job_id = build_upscale_id(request.file_id, str(source), request.scale)
        output = OUTPUT_DIR / f"{job_id}{'.png' if kind == 'image' else '.mp4'}"
        _fps = float(info.get("fps") or 0.0)
        _frames = int(info.get("frames") or 0)
        _duration = round(_frames / _fps, 3) if (_fps > 0 and _frames > 0) else None
        job = {
            "id": job_id, "object": "image.upscale" if kind == "image" else "video.upscale",
            "model": MODEL_NAME, "kind": kind,
            "file_id": request.file_id, "status": "queued", "phase": "waiting", "progress": 0.0,
            "created_at": now_iso(), "started_at": None, "finished_at": None,
            "source_provider": source_provider, "input_path": str(source), "output_path": str(output),
            "request": {
                "kind": kind, "scale": request.scale,
                "seed": request.seed if request.seed is not None else DEFAULT_SEED,
                "tile_size": request.tile_size, "vae_tile_size": request.vae_tile_size,
                "batch_size": request.batch_size, "align_method": request.align_method,
                "encoder": request.encoder, "num_gpus": request.num_gpus,
                "source_name": source.name,
                "source_width": info["width"], "source_height": info["height"],
                "source_frames": _frames, "source_fps": _fps,
                "source_duration_seconds": _duration,
                "target_width": target_width, "target_height": target_height,
            },
            "result": None, "error": None,
        }
        self.jobs[job_id] = job
        self.persist(job)
        await self.queue.put(job_id)
        logger.info("job %s queued: kind=%s | scale=%.1f | target=%dx%d", job_id, kind, request.scale, target_width, target_height)
        return self.public(job)

    def _maybe_cleanup_input(self, job: dict[str, Any]) -> None:
        if job.get("source_provider") != "upload":
            return
        input_path = job.get("input_path")
        if not input_path:
            return
        for other in self.jobs.values():
            if other.get("id") == job.get("id"):
                continue
            if other.get("input_path") == input_path and other.get("status") not in TERMINAL:
                return
        try:
            Path(input_path).unlink(missing_ok=True)
        except OSError:
            pass
        _remove_file_meta(Path(input_path).stem)

    async def run_loop(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.jobs.get(job_id)
                if not job or job.get("status") == "cancelled":
                    continue
                await self.execute(job)
            finally:
                _close_bar()
                self.queue.task_done()
                finished = self.jobs.get(job_id)
                if finished and finished.get("status") in TERMINAL:
                    self._maybe_cleanup_input(finished)
                # Runs after execute() returned, so a failed job's exception
                # traceback is already gone and the peak allocations can be
                # released back to the driver (otherwise they leak).
                reclaim_gpu_memory()

    async def execute(self, job: dict[str, Any]) -> None:
        started = time.monotonic()
        job.update(status="running", phase="preparing", progress=1.0, started_at=now_iso(), error=None)
        self.persist(job)
        request = job["request"]

        bar = _open_bar(job)
        _log_state = {"t": time.monotonic(), "p": -100.0}

        def on_progress(progress: float, phase: str) -> None:
            if job.get("status") == "cancelled":
                return  # don't let a late callback revive a cancelled job's phase
            job["progress"] = round(float(progress), 1)
            job["phase"] = phase
            now = time.monotonic()
            if bar is not None:
                # In-place progress bar (no repeated lines).
                bar.n = max(0.0, min(100.0, float(progress)))
                bar.set_postfix_str(f"{int(progress)}%, elapsed {int(now - started)}s")
                bar.refresh()
                return
            # Fallback (tqdm unavailable): periodic log lines, at least every 15s
            # or on a >=5% jump, so a long job doesn't look dead.
            if now - _log_state["t"] >= 15.0 or abs(progress - _log_state["p"]) >= 5.0:
                _log_state["t"] = now
                _log_state["p"] = progress
                logger.info("job %s %s %.1f%% (elapsed %.0fs)",
                            job["id"], phase, progress, now - started)

        def cancelled() -> bool:
            return job.get("status") == "cancelled"

        kind = str(request.get("kind", "video"))
        scale = float(request["scale"])
        seed = int(request.get("seed") if request.get("seed") is not None else DEFAULT_SEED)
        tile_size = request.get("tile_size")
        vae_tile_size = request.get("vae_tile_size")
        batch_size = request.get("batch_size")
        align_method = str(request.get("align_method", "wavelet"))
        encoder = str(request.get("encoder", "auto"))
        ngpu_req = int(request.get("num_gpus") or NUM_GPUS)
        ngpu = max(1, min(ngpu_req, max(1, len(_visible_gpus()))))

        # Single, detailed params line per job (effective tile/vae/batch/gpu shown).
        try:
            _ja, eff_tile, eff_vae, eff_batch = self.runtimes[0]._job_params(
                scale, tile_size, vae_tile_size, batch_size, align_method,
                request.get("source_width") or 1, request.get("source_height") or 1)
            eff_vae_s = "full" if not eff_vae else eff_vae
        except Exception:
            eff_tile, eff_vae_s, eff_batch = tile_size, vae_tile_size, batch_size
        logger.info(
            "job %s params: kind=%s scale=%s seed=%s align_method=%s encoder=%s num_gpus=%s | "
            "tile_size=%s vae_tile_size=%s batch_size=%s | input=%s output=%s | "
            "source=%s %sx%s frames=%s fps=%s duration=%ss -> target=%sx%s",
            job["id"], kind, scale, seed, align_method, encoder, ngpu,
            eff_tile, eff_vae_s, eff_batch, job.get("input_path"), job.get("output_path"),
            request.get("source_name"), request.get("source_width"), request.get("source_height"),
            request.get("source_frames"), request.get("source_fps"),
            request.get("source_duration_seconds"),
            request.get("target_width"), request.get("target_height"))

        try:
            if kind == "image":
                # Images are single-frame; multi-GPU does not apply.
                result = await asyncio.to_thread(
                    self.runtimes[0].upscale_image,
                    Path(job["input_path"]), Path(job["output_path"]),
                    scale, seed, tile_size, vae_tile_size, align_method, on_progress, cancelled,
                )
            elif ngpu > 1:
                # Resident model stays loaded: it serves GPU0's shard in-process,
                # GPUs 1.. reuse the persistent worker pool.
                self.runtimes[0].ensure_loaded()
                result = await asyncio.to_thread(
                    _process_multigpu,
                    ngpu, self.runtimes[0], self.pool,
                    Path(job["input_path"]), Path(job["output_path"]),
                    scale, seed, tile_size, vae_tile_size, batch_size,
                    align_method, encoder, on_progress, cancelled,
                )
            else:
                result = await asyncio.to_thread(
                    self.runtimes[0].process,
                    Path(job["input_path"]), Path(job["output_path"]),
                    scale, seed, tile_size, vae_tile_size, batch_size,
                    align_method, encoder, on_progress, cancelled,
                )
        except InferenceCancelled:
            Path(job["output_path"]).unlink(missing_ok=True)
            logger.info("job %s cancelled during execution", job["id"])
            return
        except Exception as exc:
            if job.get("status") == "cancelled":
                Path(job["output_path"]).unlink(missing_ok=True)
                logger.info("job %s cancelled during execution", job["id"])
                return
            err_msg = str(exc) or type(exc).__name__
            # The traceback keeps the failing frames' local tensors referenced
            # (asyncio's executor future holds it well past this handler), so
            # empty_cache alone leaves the peak resident. Clearing the frames
            # drops those references now, letting reclaim_gpu_memory() release
            # the memory back to the driver.
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            job.update(status="failed", phase="failed", finished_at=now_iso(),
                       error={"message": err_msg, "type": "inference_error", "param": None, "code": None})
            self.persist(job)
            logger.error("job %s failed: %s", job["id"], err_msg)
            return

        if job.get("status") == "cancelled":
            Path(job["output_path"]).unlink(missing_ok=True)
            self.persist(job)
            logger.info("job %s cancelled (post-execution)", job["id"])
            return
        if not Path(job["output_path"]).is_file():
            job.update(status="failed", phase="failed", finished_at=now_iso(),
                       error={"message": "runtime did not produce a video", "type": "server_error",
                              "param": None, "code": None})
            self.persist(job)
            logger.error("job %s failed: runtime did not produce a video", job["id"])
            return
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        job.update(status="completed", phase="completed", progress=100.0,
                   finished_at=now_iso(), result=result, error=None)
        self.persist(job)
        output_file = Path(job["output_path"])
        try:
            output_size = output_file.stat().st_size
        except OSError:
            output_size = 0
        logger.info("job %s completed: output=%s | size=%dB | elapsed=%.1fs | encoder=%s\n",
                    job["id"], output_file, output_size, result["elapsed_seconds"], result.get("encoder"))

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        if job.get("status") in TERMINAL:
            return self.public(job)
        previous_phase = job.get("phase")
        job.update(status="cancelled", phase="cancelled", finished_at=now_iso(), error=None)
        self.persist(job)
        logger.info("job %s cancelled (was %s)", job_id, previous_phase)
        if previous_phase == "waiting":
            self._maybe_cleanup_input(job)
        return self.public(job)


worker = Worker()


# --------------------------------------------------------------------------
# FastAPI 应用 / 鉴权依赖 / 文件元数据
# --------------------------------------------------------------------------
class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: Optional[str] = Field(default=None, description="通过 /v1/files 上传返回的 file_id")
    source: Optional[str] = Field(default=None, description="服务器本地图片/视频路径")
    model: Optional[str] = Field(default=None, description="模型名（可选，仅校验）")
    scale: float = Field(default=2.0, ge=1.0, le=8.0)
    seed: Optional[int] = Field(default=None,
                                description="随机种子；不填则用服务端 VOSR_SEED（默认 42）")
    tile_size: Optional[int] = Field(default=None, ge=0, le=4096)
    vae_tile_size: Optional[int] = Field(default=None, ge=0, le=8192)
    batch_size: Optional[int] = Field(default=None, ge=1, le=64)
    num_gpus: Optional[int] = Field(default=None, ge=1, le=16,
                                    description="视频帧级多卡并行的卡数；图片忽略。默认取服务端 VOSR_NUM_GPUS。")
    align_method: Literal["wavelet", "adain", "nofix"] = "wavelet"
    encoder: Literal["auto", "libx264", "nvenc"] = "auto"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("server starting on %s:%d", HOST, PORT)
    await worker.start()
    yield
    logger.info("server shutting down")
    await worker.stop()


app = FastAPI(title="VOSR2 Resident Server", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


@app.exception_handler(OpenAIHTTPException)
async def openai_exception_handler(_, exc: OpenAIHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.body})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [str(part) for part in first.get("loc", []) if part != "body"]
    return JSONResponse(status_code=422, content={"error": {
        "message": first.get("msg", "invalid request"),
        "type": "invalid_request_error",
        "param": ".".join(loc) or None,
        "code": None,
    }})


def get_upscale_or_404(upscale_id: str) -> dict[str, Any]:
    if not UPSCALE_ID_RE.fullmatch(upscale_id) or upscale_id not in worker.jobs:
        raise OpenAIHTTPException(404, "upscale not found", type="not_found_error", code="not_found")
    return worker.jobs[upscale_id]


def _file_meta_path(file_id: str) -> Path:
    return FILE_META_DIR / f"{file_id}.json"


def _write_file_meta(file_id: str, filename: str, created_at: int) -> None:
    atomic_json(_file_meta_path(file_id), {"filename": filename, "created_at": created_at})


def _read_file_meta(file_id: str) -> dict[str, Any]:
    try:
        return json.loads(_file_meta_path(file_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remove_file_meta(file_id: str) -> None:
    _file_meta_path(file_id).unlink(missing_ok=True)


def _file_object(path: Path) -> dict[str, Any]:
    stat = path.stat()
    meta = _read_file_meta(path.stem)
    return {
        "id": path.stem, "object": "file", "bytes": stat.st_size,
        "created_at": int(meta.get("created_at") or stat.st_mtime),
        "filename": meta.get("filename") or path.name, "purpose": "upscale",
    }


@app.get("/health")
async def health():
    active = sum(1 for item in worker.jobs.values() if item.get("status") == "running")
    queued = sum(1 for item in worker.jobs.values() if item.get("status") == "queued")
    return {
        "status": "ok", "service": "vosr", "model": MODEL_NAME, "mode": "resident",
        "resident": worker.runtime is not None, "active": active, "queued": queued,
        "max_queue": MAX_QUEUE, "scales": [1.0, 8.0], "auth": bool(API_KEY),
        "jobs": worker.summaries(),
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_NAME, "object": "model", "created": MODEL_CREATED, "owned_by": "vosr"}]}


@app.get("/v1/models/{model_id}", dependencies=[Depends(require_auth)])
async def get_model(model_id: str):
    if model_id != MODEL_NAME:
        raise OpenAIHTTPException(404, f"model '{model_id}' not found",
                                  type="not_found_error", param="model", code="model_not_found")
    return {"id": MODEL_NAME, "object": "model", "created": MODEL_CREATED, "owned_by": "vosr"}


@app.post("/v1/files", dependencies=[Depends(require_auth)])
async def create_file(file: UploadFile = File(...)):
    file_id = "file_" + uuid.uuid4().hex
    original_name = file.filename or "upload.mp4"
    ext = Path(original_name).suffix or ".mp4"
    dest = INPUT_DIR / f"{file_id}{ext}"
    created_at = int(time.time())
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            out.write(chunk)
    _write_file_meta(file_id, original_name, created_at)
    logger.info("uploaded file_id=%s filename=%s bytes=%d", file_id, original_name, total)
    return {"id": file_id, "object": "file", "bytes": total,
            "created_at": created_at, "filename": original_name, "purpose": "upscale"}


@app.get("/v1/files", dependencies=[Depends(require_auth)])
async def list_files():
    paths = [p for p in INPUT_DIR.glob("*") if p.is_file() and FILE_ID_RE.fullmatch(p.stem)]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"object": "list", "data": [_file_object(p) for p in paths]}


@app.delete("/v1/files/{file_id}", dependencies=[Depends(require_auth)])
async def delete_file(file_id: str):
    if not FILE_ID_RE.fullmatch(file_id):
        raise OpenAIHTTPException(404, "file not found", type="not_found_error", code="file_not_found")
    matches = list(INPUT_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise OpenAIHTTPException(404, "file not found", type="not_found_error", code="file_not_found")
    in_use = any(
        job.get("source_provider") == "upload"
        and job.get("status") not in TERMINAL
        and Path(str(job.get("input_path", ""))).stem == file_id
        for job in worker.jobs.values()
    )
    if in_use:
        raise OpenAIHTTPException(409, "file is in use by a running upscale", code="file_in_use")
    for path in matches:
        path.unlink(missing_ok=True)
    _remove_file_meta(file_id)
    logger.info("deleted file_id=%s", file_id)
    return {"id": file_id, "object": "file", "deleted": True}


# Image and video are the same endpoint (kind is auto-detected by extension),
# so the primary path is the neutral /v1/upscales; /v1/video/upscales is kept
# as a backward-compatible alias.
@app.post("/v1/upscales", status_code=202, dependencies=[Depends(require_auth)])
@app.post("/v1/video/upscales", status_code=202, dependencies=[Depends(require_auth)], include_in_schema=False)
async def create_upscale(request: UpscaleRequest):
    return await worker.submit(request)


@app.get("/v1/upscales", dependencies=[Depends(require_auth)])
@app.get("/v1/video/upscales", dependencies=[Depends(require_auth)], include_in_schema=False)
async def list_upscales():
    return {"object": "list", "data": worker.summaries()}


@app.get("/v1/upscales/{upscale_id}", dependencies=[Depends(require_auth)])
@app.get("/v1/video/upscales/{upscale_id}", dependencies=[Depends(require_auth)], include_in_schema=False)
async def get_upscale(upscale_id: str):
    return worker.public(get_upscale_or_404(upscale_id))


@app.post("/v1/upscales/{upscale_id}/cancel", dependencies=[Depends(require_auth)])
@app.post("/v1/video/upscales/{upscale_id}/cancel", dependencies=[Depends(require_auth)], include_in_schema=False)
async def cancel_upscale(upscale_id: str):
    get_upscale_or_404(upscale_id)
    return await worker.cancel(upscale_id)


@app.get("/v1/upscales/{upscale_id}/content", dependencies=[Depends(require_auth)])
@app.get("/v1/video/upscales/{upscale_id}/content", dependencies=[Depends(require_auth)], include_in_schema=False)
async def download_upscale(upscale_id: str):
    job = get_upscale_or_404(upscale_id)
    if job.get("status") != "completed":
        raise OpenAIHTTPException(409, "result is not ready", code="not_ready")
    path = Path(job["output_path"])
    if job.get("kind") == "image" or path.suffix.lower() in IMAGE_EXTS:
        media, ext = "image/png", "png"
    else:
        media, ext = "video/mp4", "mp4"
    return FileResponse(str(path), media_type=media, filename=f"vosr_{upscale_id}.{ext}")


if __name__ == "__main__":
    # CLI args override VOSR_* env vars (which override built-in defaults).
    apply_config(build_arg_parser().parse_args())
    logger.info("config: host=%s port=%d ckpt=%s work=%s output=%s dtype=%s "
                "tile=%s vae_tile=%s batch=%s seed=%s num_gpus=%s encoder=%s auth=%s",
                HOST, PORT, CKPT, WORK_DIR, OUTPUT_DIR, DTYPE,
                DEFAULT_TILE, DEFAULT_VAE_TILE, DEFAULT_BATCH, DEFAULT_SEED, NUM_GPUS,
                ENCODER, bool(API_KEY))
    uvicorn.run(app, host=HOST, port=PORT, workers=1, access_log=False,
                server_header=False, date_header=False, log_level=LOG_LEVEL.lower())

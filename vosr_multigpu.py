"""Frame-level multi-GPU video super-resolution.

Two deployment modes share one implementation:

* **CLI** (`inference_vosr_onestep_pipeline.py --num_gpus N`): one short-lived
  worker process per GPU (`python -m vosr_multigpu <cfg.json>`), concatenated at
  the end. The parent has no resident model.

* **Server** (`infer_server.py`, `num_gpus>1`): the resident model (GPU0) serves
  the GPU0 shard *in-process*, and GPUs 1..N-1 are a **persistent worker pool**
  (`python -m vosr_multigpu --server <base.json>`): each worker loads its model
  once and then serves many jobs, releasing only activations between jobs.

The shard worker is the plain function :func:`run_video_segment`; progress and
completion travel as JSON lines on stdout (workers' own logs pass through), so
nothing scrapes CLI strings or tqdm output.

Fresh worker *processes* (not in-process replicas) are used for GPUs 1..N-1 on
purpose: a clean CUDA context and guaranteed VRAM reclaim on exit.
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def build_config(**kw):
    """Fill a worker config dict with defaults. Unknown keys are kept."""
    cfg = {
        "checkpoint": None, "source": None, "output": None, "scale": 4.0,
        "align_method": "wavelet", "tile_size": None, "tile_overlap": None,
        "vae_tile_size": None, "vae_tile_overlap": None,
        "seed": 42, "frame_seed_mode": "fixed", "batch_size": 8,
        "dtype": "bf16", "codec": "mp4v", "fps": None, "num_gpus": 1,
        "start_frame": 0, "max_frames": 0, "keep_audio": True, "omp_threads": 0,
        "cuda": "0",
    }
    cfg.update(kw)
    return cfg


def _build_args(cfg):
    """Build a Namespace identical to the CLI pipeline's args for one job."""
    import vosr_core as core
    import inference_vosr_onestep_pipeline as pipe

    argv = [
        "--checkpoint", str(cfg["checkpoint"]),
        "-i", str(cfg.get("source") or "/dev/null"),
        "-o", str(cfg.get("output") or "/tmp/_vosr_unused"),
        "-u", f"{float(cfg.get('scale', 4)):g}",
        "--align_method", str(cfg.get("align_method", "wavelet")),
        "--batch_size", str(int(cfg.get("batch_size", 8))),
        "--dtype", str(cfg.get("dtype", "bf16")),
        "--infer_steps", "1",
        "--seed", str(int(cfg.get("seed", 42))),
        "--frame_seed_mode", str(cfg.get("frame_seed_mode", "fixed")),
        "--codec", str(cfg.get("codec", "mp4v")),
        "--start_frame", str(int(cfg.get("start_frame", 0))),
        "--max_frames", str(int(cfg.get("max_frames", 0))),
        "--keep_audio", "false",
        "--force_rerun",
    ]
    if cfg.get("tile_size"):
        argv += ["--tile_size", str(int(cfg["tile_size"]))]
    if cfg.get("tile_overlap"):
        argv += ["--tile_overlap", str(int(cfg["tile_overlap"]))]
    if cfg.get("vae_tile_size") is not None:
        argv += ["--vae_tile_size", str(int(cfg["vae_tile_size"]))]
    if cfg.get("vae_tile_overlap") is not None:
        argv += ["--vae_tile_overlap", str(int(cfg["vae_tile_overlap"]))]
    if cfg.get("fps"):
        argv += ["--fps", str(float(cfg["fps"]))]

    parser = pipe.build_parser()
    old_argv = sys.argv
    sys.argv = ["vosr_multigpu_worker"] + argv
    try:
        return core.load_config_with_cli(str(cfg["checkpoint"]), parser)
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# the shard worker (plain function; loads the model unless one is passed in)
# ---------------------------------------------------------------------------
def run_video_segment(cfg, progress_cb=None, models=None, device=None, cancel_check=None):
    """Upscale one contiguous frame range (cfg['start_frame'] .. +max_frames)
    into cfg['output'].

    If ``models`` is given they are reused (the caller owns them); otherwise a
    model is loaded on this process' CUDA device and freed on exit."""
    import torch                     # imported here so CUDA_VISIBLE_DEVICES,
    import vosr_core as core         # set by the caller, takes effect first.
    import inference_vosr_onestep_pipeline as pipe

    args = _build_args(cfg)
    loaded = models is None
    if loaded:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        models = core.load_models(args, device)
    try:
        pipe.process_video(str(cfg["source"]), str(cfg["output"]), args, device, models,
                           None, progress_cb, cancel_check)
    finally:
        if loaded:
            del models
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return {"output": str(cfg["output"])}


class ShardCancelled(Exception):
    """A shard stopped because its cancel predicate tripped."""


# ---------------------------------------------------------------------------
# worker process entrypoints
# ---------------------------------------------------------------------------
def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _bare_worker_logging():
    """Make this worker process log bare messages (no timestamp/level). Worker
    output is forwarded by the parent, which already adds one timestamp and an
    `[ngpu gpuN]` tag — so keeping the worker's own prefix causes double times."""
    import logging
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)


def _worker_once_main():
    """``python -m vosr_multigpu <cfg.json>`` — load, process once, exit."""
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda"])

    import vosr_core  # noqa: F401  (triggers basicConfig before we override it)
    _bare_worker_logging()

    def cb(written, total):
        if total:
            _emit({"p": round(100.0 * written / total, 2)})

    try:
        run_video_segment(cfg, cb)
        _emit({"done": True})
    except Exception:
        traceback.print_exc()
        _emit({"error": traceback.format_exc()})
        sys.exit(1)


def _worker_server_main():
    """``python -m vosr_multigpu --server <base.json>`` — load once, then serve
    JSON job lines from stdin, releasing activations (not weights) per job.

    A background thread reads stdin so a ``{"cmd":"cancel"}`` can abort the
    *current* job (raising inside ``process_video`` at the next batch boundary)
    without killing the process — the model stays loaded for the next job."""
    import queue
    import threading

    with open(sys.argv[2], "r", encoding="utf-8") as fh:
        base = json.load(fh)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(base["cuda"])

    import torch
    import vosr_core as core
    import inference_vosr_onestep_pipeline as pipe

    _bare_worker_logging()   # parent adds the timestamp/tag when forwarding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_args = _build_args(base)
    try:
        models = core.load_models(load_args, device)
    except Exception:
        traceback.print_exc()
        _emit({"error": traceback.format_exc()})
        sys.exit(1)
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    _emit({"ready": True})

    jobs: "queue.Queue" = queue.Queue()
    current: dict = {"ev": None}   # cancel Event of the in-flight job (or None)

    def _reader():
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            cmd = obj.get("cmd") if isinstance(obj, dict) else None
            if cmd == "quit":
                break
            if cmd == "cancel":
                ev = current["ev"]
                if ev is not None:
                    ev.set()
                continue
            jobs.put(obj)
        jobs.put(None)   # EOF / quit -> shut down

    threading.Thread(target=_reader, daemon=True).start()

    while True:
        job = jobs.get()
        if job is None:
            break
        ev = threading.Event()
        current["ev"] = ev

        def cb(written, total):
            if total:
                _emit({"p": round(100.0 * written / total, 2)})

        try:
            args = _build_args(job)
            pipe.process_video(str(job["source"]), str(job["output"]), args, device, models,
                               None, cb, ev.is_set)
            _emit({"done": True})
        except Exception:
            if ev.is_set():
                _emit({"cancelled": True})   # aborted; keep model loaded
            else:
                traceback.print_exc()
                _emit({"error": traceback.format_exc()})
        finally:
            current["ev"] = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# persistent worker handle + pool (server side)
# ---------------------------------------------------------------------------
class _Worker:
    def __init__(self, cuda, base_cfg, log):
        self.cuda = str(cuda)
        self.log = log
        cfg = dict(base_cfg)
        cfg["cuda"] = self.cuda
        fd, self.cfg_path = tempfile.mkstemp(prefix="vosr_worker_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda
        # Spawn only (does NOT wait for the model to finish loading) — the pool
        # starts every new worker first, then waits, so N workers load in
        # parallel across their GPUs instead of one after another.
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vosr_multigpu", "--server", self.cfg_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=BASE_DIR, env=env, start_new_session=True,
        )

    def _readline(self):
        assert self.proc.stdout is not None
        return self.proc.stdout.readline()

    def wait_ready(self):
        while True:
            line = self._readline()
            if not line:
                raise RuntimeError("worker exited before becoming ready")
            if self._handle_line(line) == "ready":
                return

    def _handle_line(self, line):
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except ValueError:
            if self.log:
                self.log(f"[ngpu gpu{self.cuda}] {line}")
            return None
        if isinstance(obj, dict):
            if obj.get("ready"):
                return "ready"
            if obj.get("done"):
                return "done"
            if obj.get("cancelled"):
                return "cancelled"
            if obj.get("error"):
                raise RuntimeError("worker error:\n" + str(obj["error"]))
            if "p" in obj:
                return float(obj["p"])
        return None

    def _send(self, obj) -> None:
        if self.proc.stdin is not None and self.proc.poll() is None:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def run(self, job_cfg, progress_cb, cancel_check):
        self._send(job_cfg)
        cancel_sent = False
        while True:
            if (cancel_check is not None and cancel_check()) and not cancel_sent:
                # Abort the current job but keep the worker (model stays loaded).
                self._send({"cmd": "cancel"})
                cancel_sent = True
            line = self._readline()
            if not line:
                raise RuntimeError("worker exited mid-job")
            val = self._handle_line(line)
            if val == "done":
                return
            if val == "cancelled":
                raise ShardCancelled()
            if isinstance(val, float) and progress_cb is not None:
                progress_cb(val, 100.0)   # (written%, total%) contract

    def alive(self):
        return self.proc.poll() is None

    def terminate(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(self.proc.pid), 15)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), 9)
            except Exception:
                pass
        try:
            os.remove(self.cfg_path)
        except OSError:
            pass


class WorkerPool:
    """Persistent per-GPU workers (loaded once, reused across requests)."""

    def __init__(self, base_cfg, devices, log=None):
        self.base_cfg = dict(base_cfg)
        self.devices = [str(d) for d in devices]
        self.log = log
        self._workers: dict[str, _Worker] = {}

    def ensure(self, n):
        """Ensure the first ``n`` workers (devices[:n]) exist and are ready.

        Missing workers are all spawned first and only then awaited, so their
        models load in parallel (one per GPU) instead of sequentially."""
        n = max(0, min(int(n), len(self.devices)))
        to_start = []
        for dev in self.devices[:n]:
            w = self._workers.get(dev)
            if w is None or not w.alive():
                if w is not None:
                    w.terminate()
                to_start.append(dev)
        if not to_start:
            return
        if self.log:
            self.log(f"[ngpu] starting persistent worker(s) on GPU {', '.join(to_start)} ...")
        started = []
        for dev in to_start:
            w = _Worker(dev, self.base_cfg, self.log)   # spawn only
            self._workers[dev] = w
            started.append(w)
        for w in started:                                # now wait, in parallel
            w.wait_ready()

    def run(self, device, job_cfg, progress_cb, cancel_check):
        dev = str(device)
        w = self._workers.get(dev)
        if w is None or not w.alive():
            raise RuntimeError(f"no live worker for GPU {dev}")
        w.run(job_cfg, progress_cb, cancel_check)

    def close(self):
        for w in list(self._workers.values()):
            w.terminate()
        self._workers.clear()


# ---------------------------------------------------------------------------
# orchestrator (executor per shard: in-process, persistent pool, or one-shot)
# ---------------------------------------------------------------------------
def _probe_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def shard_ranges(total, ngpu, first=0, max_frames=0):
    avail = max(0, total - first)
    n = min(max_frames, avail) if max_frames and max_frames > 0 else avail
    if n <= 0:
        return [], 0
    per = math.ceil(n / ngpu)
    out = []
    for g in range(ngpu):
        s = first + g * per
        c = min(per, first + n - s)
        if c <= 0:
            break
        out.append((s, c))
    return out, n


def run_multigpu(cfg, executors, progress_cb=None, cancel_check=None, log=None,
                 devices=None):
    """Shard a video across GPUs and concatenate.

    ``executors`` is a list of callables ``f(seg_cfg, progress_cb, cancel_check)``
    (one per shard, len == number of GPUs). The orchestrator runs them in threads
    and fails fast. On failure it raises RuntimeError; on cancel, ShardCancelled.
    """
    import vosr_core as core

    def _log(msg):
        (log or core.logger.info)(msg)

    source = str(cfg["source"])
    output = str(cfg["output"])
    ngpu = len(executors)
    total = _probe_frames(source)
    if total <= 0:
        raise RuntimeError(f"--num_gpus needs a known frame count, but {source} reported {total}.")

    ranges, n = shard_ranges(total, ngpu, int(cfg.get("start_frame", 0) or 0),
                             int(cfg.get("max_frames", 0) or 0))
    if not ranges:
        raise RuntimeError(f"no frames to process for {source}")
    devs = devices or list(range(ngpu))
    keep_audio = bool(cfg.get("keep_audio", True))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="vosr_ngpu_", dir=os.path.dirname(os.path.abspath(output)))
    _log(f"[ngpu] {os.path.basename(source)}: {n} frames over {len(ranges)} GPU(s) "
         f"{devs[:len(ranges)]}, ~{math.ceil(n/len(ranges))} frames/GPU")

    abort = threading.Event()

    def _cancel():
        return abort.is_set() or (cancel_check is not None and cancel_check())

    prog = [0.0] * len(ranges)
    errs: dict[int, BaseException] = {}

    def _thread(i, start, count):
        seg = os.path.join(workdir, f"seg_{i:03d}.mp4")
        seg_cfg = dict(cfg)
        seg_cfg.update({"output": seg, "start_frame": start, "max_frames": count})

        def shard_cb(written, total_frames):
            if total_frames:
                prog[i] = 100.0 * written / total_frames

        try:
            executors[i](seg_cfg, shard_cb, _cancel)
        except BaseException as exc:  # noqa: BLE001
            errs[i] = exc

    threads = [threading.Thread(target=_thread, args=(i, s, c), daemon=True)
               for i, (s, c) in enumerate(ranges)]
    for t in threads:
        t.start()

    early_cancel = False
    while any(t.is_alive() for t in threads):
        if cancel_check is not None and cancel_check():
            early_cancel = True
            abort.set()
        if errs:
            abort.set()          # fail fast: signal siblings to stop
        if progress_cb is not None:
            # prog[i] is a per-shard percentage; weight by that shard's frame
            # count so the overall bar tracks completed frames.
            frames_done = sum(prog[i] / 100.0 * ranges[i][1] for i in range(len(ranges)))
            progress_cb(min(100.0, 100.0 * frames_done / n))
        time.sleep(1.0)
    for t in threads:
        t.join()

    if errs:
        real = {i: e for i, e in errs.items() if not isinstance(e, ShardCancelled)}
        all_detail = "; ".join(f"gpu{i}={type(e).__name__}: {e!s}".strip()
                               for i, e in sorted(errs.items()))
        _log(f"[ngpu] shard failures -> {all_detail}")
        shutil.rmtree(workdir, ignore_errors=True)
        if real:
            i = min(real)
            raise RuntimeError(f"multigpu shard {i} failed:\n{real[i]}")
        raise ShardCancelled()
    if early_cancel:
        shutil.rmtree(workdir, ignore_errors=True)
        raise ShardCancelled()

    segments = [os.path.join(workdir, f"seg_{i:03d}.mp4") for i in range(len(ranges))]
    render = output + ".ngpu.mp4"
    try:
        core._concat_segments(segments, render)
        if keep_audio and core._try_mux_audio(render, source, output):
            if os.path.isfile(render):
                os.remove(render)
        else:
            os.replace(render, output)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if progress_cb is not None:
        progress_cb(100.0)
    _log(f"[ngpu] Done: {n} frames -> {output} ({len(ranges)} GPUs)")
    return {"output": output, "frames": n, "num_gpus": len(ranges)}


# ---------------------------------------------------------------------------
# executor factories
# ---------------------------------------------------------------------------
def oneshot_executor(cfg, device):
    """Short-lived ``python -m vosr_multigpu`` worker (CLI path)."""
    def run(seg_cfg, progress_cb, cancel_check):
        fd, cfg_path = tempfile.mkstemp(prefix="vosr_seg_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(seg_cfg, cuda=str(device)), fh)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        p = subprocess.Popen(
            [sys.executable, "-m", "vosr_multigpu", cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=BASE_DIR, env=env, start_new_session=True,
        )
        try:
            assert p.stdout is not None
            for line in p.stdout:
                if cancel_check is not None and cancel_check():
                    _kill(p)
                    raise ShardCancelled()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    if obj.get("error"):
                        raise RuntimeError("worker error:\n" + str(obj["error"]))
                    if "p" in obj and progress_cb is not None:
                        progress_cb(float(obj["p"]), 100.0)
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"worker exited rc={rc}")
        finally:
            if p.poll() is None:
                _kill(p)
            try:
                os.remove(cfg_path)
            except OSError:
                pass
    return run


def inproc_executor(models, device):
    """Run the shard here (server GPU0, reusing the resident model)."""
    def run(seg_cfg, progress_cb, cancel_check):
        try:
            run_video_segment(seg_cfg, progress_cb, models=models, device=device,
                              cancel_check=cancel_check)
        except Exception:
            if cancel_check is not None and cancel_check():
                raise ShardCancelled()
            raise
    return run


def pool_executor(pool, device):
    """Run the shard on a persistent pool worker."""
    def run(seg_cfg, progress_cb, cancel_check):
        pool.run(device, seg_cfg, progress_cb, cancel_check)
    return run


def _kill(p):
    try:
        os.killpg(os.getpgid(p.pid), 15)
    except Exception:
        try:
            p.terminate()
        except Exception:
            return
    try:
        p.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            pass


def is_cancelled(exc) -> bool:
    return isinstance(exc, ShardCancelled)


# ---------------------------------------------------------------------------
# CLI orchestration helper
# ---------------------------------------------------------------------------
def run_cli(cfg, ngpu, log=None):
    """One-shot-worker orchestration used by the CLI pipeline."""
    import vosr_core as core
    devices = core._visible_gpus()
    ngpu = max(1, min(int(ngpu), len(devices)))
    if ngpu <= 1:
        raise RuntimeError("run_cli needs num_gpus > 1")
    executors = [oneshot_executor(cfg, devices[g]) for g in range(ngpu)]
    return run_multigpu(cfg, executors, log=log, devices=devices[:ngpu])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--server":
        _worker_server_main()
    else:
        _worker_once_main()

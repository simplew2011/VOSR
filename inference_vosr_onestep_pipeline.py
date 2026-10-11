"""Unified image + video super-resolution with VOSR2 one-step (CLI).

Thin entrypoint over `vosr_core` (models / inference / batch upscaling / IO
helpers live there). The input kind is auto-detected by extension:
  * image (`.png/.jpg/...`)  -> single-image SR, writes a PNG.
  * video (`.mp4/.mov/...`)  -> per-frame SR, writes an MP4 (audio copied).
A folder may contain any mix of images and videos.

Video frames are processed in groups of `--batch_size` (same-size frames are
stacked into one DiT/VAE forward; ~2x at 512px, ~3.3x tiled 1024px on H800+bf16).
This is frame-by-frame *image* super-resolution, not a temporally-consistent
video model: VOSR2 has no cross-frame module.

Flicker: (1) VAE `nn.Upsample(nearest)` once returned batch-inconsistent results
on non-contiguous inputs (fixed in models/qwenimage_vae2d.py); (2) the one-step
sampler starts from random noise -- `--frame_seed_mode fixed` (default) gives
every frame the same noise, `increment` uses seed+i.

Video IO uses OpenCV (cv2). Audio is copied by default (--keep_audio is on);
pass `--keep_audio false` to skip. Copying needs `ffmpeg` on PATH.

Multi-GPU: `--num_gpus N` shards *video* frames across N GPUs, then concatenates
the segments with ffmpeg and muxes audio (images are single-frame; always run
in-process). Needs ffmpeg and GPUs visible via CUDA_VISIBLE_DEVICES.
"""
import os
import sys
import gc
import time
import argparse

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import vosr_core as core


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Source image file, video file, or a folder of images/videos (processed in '
                             'sorted order). Kind is auto-detected by extension.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Destination file or folder. A folder (or multiple inputs) writes '
                             '<stem>_<scale>x.{png|mp4} per input (by kind); a file path is only valid '
                             'for a single input.')
    parser.add_argument("--upscale", '-u', type=float, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--tile_size", type=int, default=0,
                        help='DiT tile size in pixels (0 = disabled). QUALITY knob: the DiT is trained at '
                             '512px, so use --tile_size 512 once the upscaled frame exceeds ~512px.')
    parser.add_argument("--tile_overlap", type=int, default=32,
                        help='DiT tile overlap in pixels (default 32).')
    parser.add_argument('--vae_tile_size', type=int, default=None,
                        help='VAE tile size in pixels (None/0 = full-image VAE, default and faster). Set '
                             'e.g. 1024 only when a large frame would OOM. Independent from --tile_size.')
    parser.add_argument('--vae_tile_overlap', type=int, default=None,
                        help='VAE tile overlap in pixels (default: follow --tile_overlap).')
    parser.add_argument('--posterior_mode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--infer_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frame_seed_mode', type=str, choices=['fixed', 'increment'], default='fixed',
                        help='"fixed" (default) uses one seed for all frames (less flicker); '
                             '"increment" uses seed+i per frame (image-batch semantics).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Frames per forward pass (same-size frames are batched). >1 speeds up inference '
                             '(~1.8x measured on an H800 + bf16 at 720p->1440p). Default 8; 8-16 is similar '
                             'at high resolution, and larger can regress. Requires equal frame sizes; '
                             'output is identical to batch_size 1 (per-frame fixed seeds).')
    parser.add_argument('--dtype', type=str, choices=['fp32', 'bf16'], default='fp32',
                        help='Compute dtype for DiT and DINOv2 (VAE stays fp32). bf16 is ~2x faster.')
    parser.add_argument('--fps', type=float, default=None, help='Output FPS (default: source FPS).')
    parser.add_argument('--start_frame', type=int, default=0, help='First source frame index to process.')
    parser.add_argument('--max_frames', type=int, default=0, help='Max frames to process (0 = all).')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='OpenCV fourcc for the output video (default mp4v; try avc1/H264 if available).')
    parser.add_argument('--save_frames', type=str, default=None,
                        help='If set, also save each SR frame as PNG into this directory.')
    parser.add_argument('--keep_audio', type=_str2bool, nargs='?', const=True, default=True,
                        metavar='BOOL',
                        help='Copy the source audio track onto the output (on by default; requires an '
                             'ffmpeg binary on PATH). Pass --keep_audio false to disable.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Shard the video across this many GPUs (frame-level data parallelism). Each GPU '
                             'runs a full replica on a contiguous frame range; outputs are concatenated. '
                             'Measured ~1.8x (2 GPU) / ~2.9x (4) / ~3.9x (8) at 720p->1440p. Needs ffmpeg '
                             'for concatenation. Uses the GPUs visible via CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--omp_threads', type=int, default=0,
                        help='Per-process CPU threads for --num_gpus workers (0 = auto, roughly '
                             'cpu_count/(2*N) capped at 8) to avoid oversubscription across replicas.')
    parser.add_argument('--force_rerun', action='store_true')
    return parser


def _process_multigpu(src, out_path, args):
    """Frame-level data-parallel video SR: shard frames across GPUs, then concatenate.

    Thin wrapper over ``vosr_multigpu.run_multigpu`` (the shared implementation,
    also used by the HTTP server). Returns True on success, False when there is
    not more than one visible GPU (caller falls back to the single-process path).
    """
    if os.path.isfile(out_path) and not args.force_rerun:
        core.logger.info("Output %s already exists (use --force_rerun to overwrite). Skip.", out_path)
        return True

    devs = core._visible_gpus()
    ngpu = min(int(args.num_gpus), len(devs))
    if ngpu <= 1:
        core.logger.warning("Only %d GPU visible; falling back to single-process path.", len(devs))
        return False  # caller falls back

    total = int(cv2.VideoCapture(src).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, int(args.start_frame))
    if total <= 0:
        raise RuntimeError(
            f"--num_gpus needs a known frame count, but {src} reported {total}. "
            f"Run without --num_gpus for streams whose length is unknown.")

    import vosr_multigpu as mg
    cfg = mg.build_config(
        checkpoint=args.checkpoint, source=src, output=out_path, scale=args.upscale,
        align_method=args.align_method, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        vae_tile_size=args.vae_tile_size, vae_tile_overlap=args.vae_tile_overlap,
        seed=args.seed, frame_seed_mode=args.frame_seed_mode, batch_size=args.batch_size,
        dtype=args.dtype, codec=args.codec, fps=args.fps, num_gpus=ngpu,
        start_frame=start, max_frames=args.max_frames, keep_audio=bool(args.keep_audio),
        omp_threads=args.omp_threads,
    )
    mg.run_cli(cfg, ngpu, log=core.logger.info)
    return True


def _input_kind(path):
    """'image' | 'video' | None, by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in core.IMG_EXTENSIONS:
        return "image"
    if ext in core.VIDEO_EXTENSIONS:
        return "video"
    return None


def _list_inputs(path):
    """A file -> [path]; a folder -> every image/video file in it (sorted)."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if _input_kind(f) is not None
        )
    raise FileNotFoundError(f"Input file/folder not found: {path}")


def _resolve_outputs(inputs, output, upscale):
    """Map each input to an output path.

    A folder (or multiple inputs) writes `<stem>_<scale>x.png` for images and
    `<stem>_<scale>x.mp4` for videos; a single file path is used verbatim.
    """
    as_dir = (
        len(inputs) > 1
        or os.path.isdir(output)
        or output.endswith(("/", os.sep))
    )
    if as_dir:
        os.makedirs(output, exist_ok=True)
        outs = []
        for src in inputs:
            stem = os.path.splitext(os.path.basename(src))[0]
            ext = ".png" if _input_kind(src) == "image" else ".mp4"
            outs.append(os.path.join(output, f"{stem}_{upscale:g}x{ext}"))
        return outs
    if _input_kind(output) is None:
        base, ext = os.path.splitext(output)
        output = output + (".png" if _input_kind(inputs[0]) == "image" else ".mp4")
    return [output]


def process_image(src, out_path, args, device, models):
    """Run VOSR2 on one image and save the PNG at out_path."""
    vae, venc, model, vosr_model, light_decoder = models
    sample_fn = vosr_model.sample_onestep

    if os.path.isfile(out_path) and not args.force_rerun:
        core.logger.info("Output %s already exists (use --force_rerun to overwrite). Skip.", out_path)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    raw_img = Image.open(src).convert("RGB")
    w, h = raw_img.size
    target_w, target_h = int(round(w * args.upscale)), int(round(h * args.upscale))
    core.logger.info("Source: %s  %dx%d -> %dx%d", src, w, h, target_w, target_h)
    core._print_tiling_advice(args, target_w, target_h)

    input_img = raw_img.resize((target_w, target_h), Image.BICUBIC)
    lq = transforms.ToTensor()(input_img).unsqueeze(0).to(device) * 2.0 - 1.0
    torch.manual_seed(args.seed)

    with torch.no_grad():
        if args.tile_size > 0:
            sr_tensor = core.tiled_latent_inference(
                model, vae, venc, lq, args, device=device, light_decoder=light_decoder,
            )
        else:
            lq_latent, latents_mean, latents_std = core.encode_dispatch(vae, lq, args, device)
            z_fea = core.get_venc_features(venc, lq, args)
            mdt = next(model.parameters()).dtype
            lq_latent = lq_latent.to(mdt)
            z_fea = [f.to(mdt) for f in z_fea]
            sr_latent = sample_fn(model, lq_latent, n_steps=args.infer_steps, venc_fea=z_fea)
            sr_tensor = core.decode_dispatch(vae, sr_latent, args, latents_mean, latents_std, light_decoder)

    sr_img = transforms.ToPILImage()(sr_tensor[0].cpu() * 0.5 + 0.5)
    if args.align_method == 'adain':
        sr_img = core.adain_color_fix(sr_img, input_img)
    elif args.align_method == 'wavelet':
        sr_img = core.wavelet_color_fix(sr_img, input_img)
    sr_img.save(out_path)
    core.logger.info("Done: 1 image -> %s (%dx%d)", out_path, target_w, target_h)


def process_video(src, out_path, args, device, models, save_frames_dir, progress_cb=None,
                  cancel_check=None):
    """Run VOSR2 per-frame over one video and write the result to out_path.

    progress_cb(written_frames, total_frames) is called after every batch when
    given (used by the multi-GPU worker to report progress back to the server).
    cancel_check() -> bool aborts between batches (raises RuntimeError)."""
    vae, venc, model, vosr_model, light_decoder = models
    sample_fn = vosr_model.sample_onestep

    if os.path.isfile(out_path) and not args.force_rerun:
        core.logger.info("Output %s already exists (use --force_rerun to overwrite). Skip.", out_path)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = float(args.fps) if args.fps else (src_fps if src_fps > 0 else 25.0)
    core.logger.info("Source: %s  fps=%.3f  frames=%d (reported)", src, src_fps, src_total)

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    want_audio = bool(args.keep_audio)
    render_path = (out_path + ".video-only.mp4") if want_audio else out_path

    writer = None
    out_w = out_h = None
    written = 0
    errors = 0
    advised = False
    t_start = time.time()

    n_expected = None
    if args.max_frames and args.max_frames > 0:
        n_expected = args.max_frames
    elif src_total > args.start_frame:
        n_expected = src_total - args.start_frame
    pbar = tqdm(total=n_expected, file=sys.stderr, desc="frames", unit="f",
                disable=progress_cb is not None)

    pending = []          # list of (index, bgr frame)
    pending_seeds = []

    def flush():
        nonlocal writer, out_w, out_h, written, errors, advised
        if not pending:
            return
        frames = [f for _, f in pending]
        try:
            results = core.upscale_frames_batch(
                frames, model, vae, venc, sample_fn, args, device, light_decoder, list(pending_seeds),
            )
        except Exception as e:
            # Fail loudly instead of silently substituting bicubic: a silent
            # fallback made OOM'd batches look "completed" but output bicubic
            # (not VOSR), which is impossible to notice downstream.
            errors += len(frames)
            core.logger.error("Batch of %d frame(s) failed: %s", len(frames), e)
            pending.clear()
            pending_seeds.clear()
            raise

        for (frame_idx, _), (out_bgr, sr_pil) in zip(pending, results):
            if not advised:
                out_h, out_w = out_bgr.shape[:2]
                if not args.tile_size or args.tile_size <= 0:
                    core._print_tiling_advice(args, out_w, out_h)
                core.logger.info("Output: %dx%d @ %.3f fps, codec=%s", out_w, out_h, out_fps, args.codec)
                writer = cv2.VideoWriter(render_path, cv2.VideoWriter_fourcc(*args.codec), out_fps, (out_w, out_h))
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV VideoWriter could not open {render_path} with codec {args.codec!r}")
                advised = True
            writer.write(out_bgr)
            if save_frames_dir:
                sr_pil.save(os.path.join(save_frames_dir, "frame_%06d.png" % frame_idx))
            written += 1

        pbar.update(len(frames))
        pbar.set_postfix_str("batch=%d" % len(frames), refresh=False)
        if progress_cb is not None:
            progress_cb(written, n_expected or src_total)

        pending.clear()
        pending_seeds.clear()
        if written % 200 < len(frames):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    def frame_seed(global_index):
        return args.seed if args.frame_seed_mode == 'fixed' else args.seed + global_index

    idx = -1
    try:
        while True:
            if cancel_check is not None and cancel_check():
                raise RuntimeError("cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx < args.start_frame:
                continue
            if args.max_frames and args.max_frames > 0 and written + len(pending) >= args.max_frames:
                break

            if pending and frame.shape != pending[0][1].shape:
                flush()  # resolution changed mid-stream; process what we have first

            pending.append((idx, frame))
            pending_seeds.append(frame_seed(idx))
            if len(pending) >= args.batch_size:
                flush()
    finally:
        try:
            flush()
        finally:
            pbar.close()
            cap.release()
            if writer is not None:
                writer.release()

    if written == 0:
        core.logger.warning("No frames were processed. Nothing written for %s.", src)
        return

    if want_audio:
        if core._try_mux_audio(render_path, src, out_path):
            os.remove(render_path)
            core.logger.info("Audio copied from %s.", src)
        else:
            os.replace(render_path, out_path)

    elapsed = time.time() - t_start
    core.logger.info(
        "Done: %d frames -> %s (%dx%d @ %.3f fps) in %.1f min (%.2f fps, %d frame errors)",
        written, out_path, out_w, out_h, out_fps, elapsed / 60.0, written / max(elapsed, 1e-6), errors,
    )


def main():
    parser = build_parser()
    temp_args, _ = parser.parse_known_args()
    args = core.load_config_with_cli(temp_args.checkpoint, parser)

    if args.infer_steps is None:
        args.infer_steps = 1
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)
    args.batch_size = max(1, int(args.batch_size))

    inputs = _list_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No images/videos found at: {args.input}")
    outputs = _resolve_outputs(inputs, args.output, args.upscale)
    for src, dst in zip(inputs, outputs):
        core.logger.info("Queued: %s -> %s (%s)", src, dst, _input_kind(src))

    # Lazily load an in-process model only when something actually needs it
    # (images, or videos when multi-GPU sharding is not used/available).
    local = {"models": None, "device": None}

    def local_models():
        if local["models"] is None:
            local["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            core.logger.info(
                "Loading model from %s ... (steps=%d, seed=%d, seed_mode=%s, dtype=%s, batch=%d)",
                args.checkpoint, args.infer_steps, args.seed, args.frame_seed_mode, args.dtype, args.batch_size)
            local["models"] = core.load_models(args, local["device"])
        return local["device"], local["models"]

    use_ngpu = int(getattr(args, 'num_gpus', 1) or 1) > 1
    for src, dst in zip(inputs, outputs):
        kind = _input_kind(src)
        # Frame-level data parallelism applies to videos only (images are single
        # frame). It spawns worker processes, so the parent must not load a model.
        if kind == "video" and use_ngpu and _process_multigpu(src, dst, args) is not False:
            continue
        device, models = local_models()
        if kind == "image":
            process_image(src, dst, args, device, models)
        else:
            save_frames_dir = args.save_frames
            if save_frames_dir and len(inputs) > 1:
                save_frames_dir = os.path.join(save_frames_dir, os.path.splitext(os.path.basename(src))[0])
            process_video(src, dst, args, device, models, save_frames_dir)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

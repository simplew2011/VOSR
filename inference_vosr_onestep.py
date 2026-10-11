import os
import sys
import gc
import types
import glob
import argparse
import json
import math
import logging
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize
from safetensors.torch import load_file
from tqdm import tqdm

# The vendored DINOv2 prints "xFormers is not available" on import and falls back
# to PyTorch SDPA (same class of kernel, ~no slowdown). Silence the noise.
warnings.filterwarnings("ignore", message=r".*xFormers is (not )?available.*")
warnings.filterwarnings("ignore", message=r".*xFormers is (not )?disabled.*")

logging.basicConfig(
    level=logging.INFO,
    format='[\033[34m%(asctime)s\033[0m] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("VOSR")

torch.hub.set_dir('preset/ckpts/torch_cache')
sys.path.append(os.getcwd())
# Make the VOSR/ dir (where tiled_vae.py lives) importable from any run cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_sageattention_importable():
    """Guard against a broken `sageattention` install blocking `import diffusers`.

    A shared/relocated conda env may ship a `sageattention` built for a different
    CUDA runtime (e.g. it links `libcudart.so.13`), which raises ImportError on
    this machine. `diffusers` imports sageattention eagerly, so that broken build
    makes `import diffusers` (and therefore this whole pipeline) fail. If the real
    import fails, register a harmless stub so downstream imports succeed; VOSR uses
    the default PyTorch SDPA attention backend, so the stub's functions are never
    called (they raise a clear error if they ever are).
    """
    try:
        import sageattention  # noqa: F401
        return
    except Exception:
        pass
    import importlib.machinery as _machinery

    _names = (
        "sageattn", "sageattn_varlen",
        "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton",
        "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_sm90",
    )

    def _unavailable(*_a, **_k):
        raise RuntimeError(
            "sageattention is installed but not importable in this environment "
            "(likely built against a different CUDA runtime). Use the SDPA "
            "attention backend, or fix/reinstall sageattention.")

    _stub = types.ModuleType("sageattention")
    for _n in _names:
        setattr(_stub, _n, _unavailable)
    _stub.__version__ = "2.2.0"  # >= diffusers' required 2.1.1
    _stub.__spec__ = _machinery.ModuleSpec("sageattention", loader=None)
    sys.modules["sageattention"] = _stub
    logger.warning(
        "sageattention failed to import (broken env build); using a stub so diffusers "
        "imports. VOSR attention falls back to PyTorch SDPA.")


_ensure_sageattention_importable()

from models.lightningdit import LightningDiT
from models.light_decoder import LightDecoder
from vosr import VOSR
from tiled_vae import (
    encode_dispatch, decode_dispatch,
    _gaussian_weights, _make_tile_grid, _log_once,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

SD2_LWDECODER_PATH = 'preset/ckpts/sd21_lwdecoder.pth'
SD2_AE_PATH = 'preset/ckpts/stable-diffusion-2-1-base'
QWEN_AE_PATH = 'preset/ckpts/Qwen-Image-vae-2d'


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_lq_images(input_path):
    p = os.path.normpath(input_path)
    if os.path.isfile(p):
        return [p]
    if os.path.isdir(p):
        imgs = [
            os.path.join(p, f) for f in sorted(os.listdir(p))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        return imgs
    return []


def all_expected_outputs_exist(image_paths, out_dirs):
    if not image_paths:
        return False
    for img_path in image_paths:
        name = os.path.basename(img_path)
        for d in out_dirs:
            if not os.path.isfile(os.path.join(d, os.path.splitext(name)[0] + '.png')):
                return False
    return True


def adain_color_fix(target, source):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    eps = 1e-5
    target_mean = torch.mean(target_tensor, dim=[2, 3], keepdim=True)
    target_std = torch.std(target_tensor, dim=[2, 3], keepdim=True) + eps
    source_mean = torch.mean(source_tensor, dim=[2, 3], keepdim=True)
    source_std = torch.std(source_tensor, dim=[2, 3], keepdim=True) + eps
    target_tensor = (target_tensor - target_mean) / target_std * source_std + source_mean
    return to_pil_image(torch.clamp(target_tensor[0], 0, 1))


def wavelet_color_fix(target, source):
    import cv2
    target_np = np.array(target).astype(np.float32) / 255.0
    source_np = np.array(source.resize(target.size, Image.LANCZOS)).astype(np.float32) / 255.0
    sigma = 5
    source_low = cv2.GaussianBlur(source_np, (0, 0), sigma)
    target_low = cv2.GaussianBlur(target_np, (0, 0), sigma)
    target_high = target_np - target_low
    result = np.clip(source_low + target_high, 0, 1) * 255.0
    return Image.fromarray(result.astype(np.uint8))


def load_config_with_cli(ckpt_path, parser):
    cur = os.path.abspath(ckpt_path)
    json_path = None
    for _ in range(5):
        candidate = os.path.join(cur, "args.json")
        if os.path.isfile(candidate):
            json_path = candidate
            break
        cur = os.path.dirname(cur)
    if json_path is None:
        logger.warning("args.json not found near %s, using default CLI args only.", ckpt_path)
        return parser.parse_args()
    with open(json_path, 'r') as f:
        json_args = json.load(f)
    cli_args = parser.parse_args()
    combined_args = argparse.Namespace()
    for k, v in json_args.items():
        setattr(combined_args, k, v)
    for k, v in vars(cli_args).items():
        if v is not None:
            setattr(combined_args, k, v)
        elif not hasattr(combined_args, k):
            setattr(combined_args, k, v)
    return combined_args


def _clear_stale_distributed_env():
    for k in (
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
        'GROUP_RANK', 'ROLE_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_MAX_RESTARTS',
    ):
        os.environ.pop(k, None)


def _inference_accelerator_stub():
    _clear_stale_distributed_env()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return types.SimpleNamespace(device=dev)


def preprocess_raw_image(x, args):
    x = x / 255.
    x = F.interpolate(x, args.dinov2_size, mode='bicubic').clip(0., 1.)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


def _hub_load_dinov2(model_name):
    """Load a DINOv2 backbone. Prefer the local torch.hub cache when present so
    offline compute nodes never block on a github probe (torch.hub only catches
    URLError when resolving the default branch, so a hung/reset connection can
    abort or stall the cache path)."""
    local = os.path.join('preset', 'ckpts', 'torch_cache',
                         'facebookresearch_dinov2_main')
    if os.path.isdir(local):
        try:
            return torch.hub.load(local, model_name, source='local')
        except Exception:
            pass
    try:
        return torch.hub.load('facebookresearch/dinov2', model_name)
    except Exception:
        if os.path.isdir(local):
            return torch.hub.load(local, model_name, source='local')
        raise


def load_dinov2(args, device):
    if args.enc_type == 'dinov2b':
        encoder = _hub_load_dinov2('dinov2_vitb14')
    elif args.enc_type == 'dinov2l':
        encoder = _hub_load_dinov2('dinov2_vitl14')
    elif args.enc_type == 'dinov2g':
        encoder = _hub_load_dinov2('dinov2_vitg14')
    del encoder.head
    encoder.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        features = {}
        layer_indices = list(range(len(self.blocks)))
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features[f'layer_{i}'] = x[:, 1:]
        x_norm = self.norm(x)
        return features, x_norm[:, 1:]

    encoder.forward_with_features = types.MethodType(forward_with_features, encoder)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_venc_features(venc, lq_tensor, args):
    with torch.no_grad():
        raw_image = (0.5 * lq_tensor + 0.5) * 255
        raw_image_ = preprocess_raw_image(raw_image, args)
        raw_image_ = raw_image_.to(next(venc.parameters()).dtype)
        features, x_norm = venc.forward_with_features(raw_image_)
        z = [v for k, v in features.items() if k.startswith('layer_')]
        z[-1] = x_norm
        z = [z[i] for i in args.layer_dinov2b_list]
    return z


class SquarePadDiT(torch.nn.Module):
    """Wrap the DiT so non-square latents work.

    LightningDiT.forward_flexible only supports square latents (H == W). For
    non-square inputs we reflect-pad the latent to a square (rounded up to a
    multiple of patch_size), run the model, then crop back to the original
    size. This mirrors TE-Speed-VOSR2's backend `_dit_call`. It is a no-op for
    square inputs, so square-image behaviour is unchanged.
    """

    def __init__(self, model, patch_size=2):
        super().__init__()
        self.model = model
        self.patch_size = max(1, int(patch_size))

    def forward(self, x, t, r=None, z=None):
        _, _, h, w = x.shape
        if h == w:
            return self.model(x, t, r, z)
        p = self.patch_size
        side = ((max(h, w) + p - 1) // p) * p
        ph, pw = side - h, side - w
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        out = self.model(xp, t, r, z)
        return out[:, :, :h, :w]


def _crop_venc_features(z_fea_full, hi, wi, he, we, lh, lw):
    """Crop DINOv2 feature list from latent-space coordinates."""
    tiles = []
    for z_layer in z_fea_full:
        B, N, C = z_layer.shape
        fs = int(math.sqrt(N))
        fh_s = int(hi / lh * fs)
        fh_e = max(int(he / lh * fs), fh_s + 1)
        fw_s = int(wi / lw * fs)
        fw_e = max(int(we / lw * fs), fw_s + 1)
        z_2d = z_layer.view(B, fs, fs, C)
        tiles.append(z_2d[:, fh_s:fh_e, fw_s:fw_e, :].reshape(B, -1, C))
    return tiles


def tiled_latent_inference(
    model, vae, venc, lq_tensor, args,
    device='cuda', light_decoder=None, noise_fn=None,
):
    """
    Latent-space tiled inference for VOSR DiT.

    1. VAE-encode the full image once (no tile seams from encoder).
    2. For each tile, crop the corresponding LQ pixel region, resize to
       dinov2_size, and extract DINOv2 features independently.  This keeps
       exact spatial correspondence (same as training) while producing a
       constant feature shape across tiles (avoids @torch.compile recompilation).
    3. Generate noise z for the full latent (consistent across overlapping tiles).
    4. For each flow step, tile only the DiT forward pass in latent space,
       blend velocity predictions with Gaussian weights, then update z globally.
    5. VAE-decode the full result latent once.
    """
    AE_FACTOR = 8
    b = lq_tensor.shape[0]
    patch_size = getattr(args, 'patch_size', 2)

    # --- Full-image encode (done once) ---
    with torch.no_grad():
        lq_latent, latents_mean, latents_std = encode_dispatch(vae, lq_tensor, args, device)

    lq_latent = lq_latent.to(next(model.parameters()).dtype)
    _, lc, lh, lw = lq_latent.shape

    # --- Pixel tile params -> latent tile params (aligned to patch_size) ---
    lt_size = max((args.tile_size // AE_FACTOR // patch_size) * patch_size, patch_size)
    lt_overlap = max(args.tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)

    # --- Fast path: no tiling needed ---
    if lh <= lt_size and lw <= lt_size:
        _log_once(f"[Tiled Latent]: latent {lh}x{lw} fits in tile {lt_size}, no tiling needed.")
        with torch.no_grad():
            z_fea = get_venc_features(venc, lq_tensor, args) if venc is not None else None
            z = noise_fn(lq_latent.shape).to(lq_latent.dtype) if noise_fn is not None else torch.randn_like(lq_latent)
            n_steps = args.infer_steps
            t_seq = torch.linspace(1., 0., n_steps + 1, device=device)
            for i in range(n_steps):
                t_cur, t_nxt = t_seq[i], t_seq[i + 1]
                u = model(torch.cat([lq_latent, z], 1),
                          t_cur.expand(b), t_nxt.expand(b), z_fea)
                z = z - (t_cur - t_nxt) * u
            return decode_dispatch(vae, z, args, latents_mean, latents_std, light_decoder)

    # --- Build tile grid & Gaussian weights ---
    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
    g_weight = _gaussian_weights(lt_size, lt_size, lc, device)
    _log_once(f"[Tiled Latent]: latent {lh}x{lw}, tile={lt_size}, overlap={lt_overlap}, "
              f"grid={len(h_pos)}x{len(w_pos)}")

    # --- Per-tile DINOv2 features (pre-computed once) ---
    # Each tile's pixel crop is independently resized to dinov2_size and fed
    # through DINOv2, producing features with constant shape that exactly match
    # the tile's spatial content (same as training-time behaviour).
    tile_venc = {}
    if venc is not None:
        with torch.no_grad():
            for hi in h_pos:
                for wi in w_pos:
                    ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                    ph_e = min((hi + lt_size) * AE_FACTOR, lq_tensor.shape[2])
                    pw_e = min((wi + lt_size) * AE_FACTOR, lq_tensor.shape[3])
                    lq_crop = lq_tensor[:, :, ph_s:ph_e, pw_s:pw_e]
                    tile_venc[(hi, wi)] = get_venc_features(venc, lq_crop, args)

    # --- Full-latent noise (shared across tiles for consistency) ---
    z = noise_fn(lq_latent.shape).to(lq_latent.dtype) if noise_fn is not None else torch.randn_like(lq_latent)

    # --- Flow matching loop with tiled DiT ---
    n_steps = args.infer_steps
    t_seq = torch.linspace(1., 0., n_steps + 1, device=device)

    with torch.no_grad():
        for step_i in range(n_steps):
            t_cur, t_nxt = t_seq[step_i], t_seq[step_i + 1]
            dt = t_cur - t_nxt

            u_acc = torch.zeros_like(lq_latent)
            w_acc = torch.zeros_like(lq_latent)

            for hi in h_pos:
                for wi in w_pos:
                    he, we = hi + lt_size, wi + lt_size

                    inp = torch.cat([lq_latent[:, :, hi:he, wi:we],
                                     z[:, :, hi:he, wi:we]], dim=1)

                    z_fea_tile = tile_venc.get((hi, wi))
                    u_tile = model(inp, t_cur.expand(b), t_nxt.expand(b), z_fea_tile)

                    u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                    w_acc[:, :, hi:he, wi:we] += g_weight

            z = z - dt * (u_acc / w_acc)

    # --- Full-latent decode (done once) ---
    with torch.no_grad():
        return decode_dispatch(vae, z, args, latents_mean, latents_std, light_decoder)


def _estimate_vae_peak_gb(w, h):
    """Rough peak-memory estimate (GB) for full-image Qwen VAE encode+decode.

    Measured on an H800 with the fp32 Qwen VAE:
      - activations grow ~2.2 GB per output megapixel (valid 0.3..11 Mpix);
      - above ~11 Mpix the VAE's single-head attention can fall back to an
        O(N^2) kernel where N = (w/8)*(h/8) = pixels/64 tokens, allocating an
        N x N matrix -- this is what actually OOMs (e.g. 3584x3584 tried to
        allocate a single 82.7 GB tensor).
    The estimate therefore takes the larger of the two terms.
    """
    mpix = (w * h) / 1e6
    linear = 2.2 * mpix + 0.2
    tokens = (w * h) / 64.0                 # latent tokens (1/8 x 1/8 spatial)
    attn = (tokens ** 2) * 2.0 / 1e9        # worst-case fp16 attention matrix
    return max(linear, attn)


def _print_tiling_advice(args, w, h):
    """Advise on tiling once the processed image exceeds the model's training size.

    The two tiling switches are independent:
      --tile_size 512       DiT tiling -> QUALITY. The DiT is trained at
                            `args.resolution` (512px); running the full latent at a
                            larger size is off-distribution and looks blurry/flat
                            (cswry/VOSR issue #10). Recommended once the processed
                            (post-upscale) image exceeds ~512px.
      --vae_tile_size 1024  VAE tiling -> MEMORY only. The Qwen VAE attention can
                            fall back to O(N^2) memory on large inputs, so this is
                            only needed when a full-image encode/decode would OOM.
                            It barely changes the result (Gaussian-blended).
    See README "Tiled inference for large images" and issues #10 / #4.
    """
    train = int(getattr(args, 'resolution', 512) or 512)
    long_side = max(w, h)
    if long_side <= train * 1.25:
        return
    if args.tile_size <= 0:
        logger.warning(
            "Processed image is %dx%d (~%.1fx the %dpx training resolution) and "
            "--tile_size is not set.\n"
            "  The DiT is trained at %dx%d; full-image inference at this size is "
            "off-distribution and typically gives blurry/flat results or artifacts "
            "(cswry/VOSR issue #10).\n"
            "  Recommended: add  --tile_size %d  (the VAE does not need tiling at this size)",
            w, h, long_side / train, train, train, train, train,
        )
    else:
        if args.tile_size != train:
            logger.warning(
                "--tile_size %d differs from the %dpx training resolution; "
                "--tile_size %d usually gives the best quality.",
                args.tile_size, train, train,
            )
        if args.tile_overlap < 16:
            logger.warning(
                "--tile_overlap %d is small; large images can show seams or "
                "random-colored tiles (cswry/VOSR issue #4 used 16). Prefer the default 32 or larger.",
                args.tile_overlap,
            )
    # VAE tiling is a separate, memory-only knob. Estimate the full-image VAE
    # peak and compare against the free GPU memory to decide whether it is needed.
    if not getattr(args, 'vae_tile_size', None):
        est = _estimate_vae_peak_gb(w, h)
        free_gb = None
        if torch.cuda.is_available():
            try:
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
            except Exception:
                free_gb = None
        if free_gb is None or est > 0.85 * free_gb:
            extra = "" if free_gb is None else " (only %.1f GB free)" % free_gb
            logger.warning(
                "Full-image VAE encode/decode of %dx%d is estimated to peak at ~%.1f GB%s; "
                "this may run out of memory.\n"
                "  Consider  --vae_tile_size 1024  (memory-only knob; barely changes the result).",
                w, h, est, extra,
            )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument("--upscale", '-u', type=float, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--tile_size", type=int, default=0,
                        help='DiT tile size in pixels (default 0 = disabled). This is the QUALITY knob: '
                             'the DiT is trained at 512px, so once the processed (post-upscale) image '
                             'exceeds ~512px, use --tile_size 512 for good results (overlapping tiles are '
                             'Gaussian-blended). Full-image inference at large sizes is off-distribution '
                             'and looks blurry/flat. Independent from --vae_tile_size.')
    parser.add_argument("--tile_overlap", type=int, default=32,
                        help='DiT tile overlap in pixels (default 32). Too small (e.g. <16) can leave '
                             'seams or random-colored tiles on large images (see cswry/VOSR issue #4).')
    parser.add_argument('--vae_tile_size', type=int, default=None,
                        help='VAE tile size in pixels for tiled encode/decode (default 0 = full-image VAE). '
                             'This is a MEMORY knob, not a quality one: the Qwen VAE attention is O(N)-memory '
                             'and fits a full image up to ~8K, so only set it (e.g. 1024) when a very large '
                             'output runs out of memory. Independent from --tile_size.')
    parser.add_argument('--vae_tile_overlap', type=int, default=None,
                        help='VAE tile overlap in pixels (default: follow --tile_overlap).')
    parser.add_argument('--posterior_mode', action=argparse.BooleanOptionalAction, default=True,
                        help='Deterministic VAE latent (latent_dist.mode()) instead of .sample() '
                             '(needed for seam-free tiled encode).')
    parser.add_argument('--infer_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, choices=['fp32', 'bf16'], default='fp32',
                        help='Compute dtype for the DiT and DINOv2 encoder. bf16 is ~2x faster '
                             'and roughly halves peak memory; the Qwen VAE stays fp32.')
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--config', type=str, default='')
    return parser


def load_models(args, device):
    """Build the VAE, vision encoder, DiT and the VOSR sampler.

    Shared by the image (this file) entry points. Returns (vae, venc, model, vosr_model, light_decoder).
    """
    # 1. Load VAE
    ae_path = getattr(args, 'ae_path', QWEN_AE_PATH if args.ae_type == 'qwen' else SD2_AE_PATH)
    if args.ae_type == "qwen":
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(ae_path)
    elif args.ae_type == "sd2":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(ae_path, subfolder="vae")
    vae.to(device).eval()

    # 2. Load LightDecoder (sd2 only)
    light_decoder = None
    if args.ae_type == 'sd2':
        logger.info("Loading LightDecoder from %s...", SD2_LWDECODER_PATH)
        ckpt = torch.load(SD2_LWDECODER_PATH, map_location="cpu")
        dec_config = ckpt["config"]
        light_decoder = LightDecoder(
            in_channels=dec_config["in_channels"],
            out_channels=dec_config["out_channels"],
            block_out_channels=tuple(dec_config["block_out_channels"]),
            layers_per_block=dec_config["layers_per_block"],
        )
        light_decoder.load_state_dict(ckpt["model_state_dict"])
        light_decoder.to(device).eval()

    # 3. Load Vision Encoder (dinov2)
    venc = load_dinov2(args, device)

    # 4. Load DiT Model (distilled student with auxiliary_time_cond)
    base_channel = 4 if args.ae_type == 'sd2' else 16
    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2 * base_channel,
        out_channels=base_channel,
        hidden_size=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio,
        auxiliary_time_cond=args.auxiliary_time_cond,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list),
    )

    search_dirs = [
        os.path.join(args.checkpoint, "clean_weights"),
        os.path.join(args.checkpoint, "checkpoints"),
        args.checkpoint,
    ]
    weight_path = None
    for d in search_dirs:
        for name in ("ema_model.safetensors", "model.safetensors"):
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                weight_path = candidate
                break
        if weight_path:
            break
    if weight_path is None:
        hits = glob.glob(os.path.join(args.checkpoint, "**/*.safetensors"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"No .safetensors found under {args.checkpoint}")
        weight_path = hits[0]

    logger.info("Loading weights from %s", weight_path)
    state_dict = load_file(weight_path) if weight_path.endswith(".safetensors") else torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    model.forward = model.forward_flexible

    run_dtype = torch.bfloat16 if getattr(args, 'dtype', 'fp32') == 'bf16' else torch.float32
    if run_dtype == torch.bfloat16:
        # Explicit weight cast (not torch.autocast): measured ~3% faster here, and
        # identical quality. The known autocast-motivating pitfall -- TimestepEmbedder
        # builds t_freq in fp32 then F.linear(bf16_w, fp32_in) mismatches -- is already
        # handled at its source in models/lightningdit.py (t_freq.to(weight.dtype)), so
        # plain autoregressive casts (lq_latent / z_fea / raw_image) suffice. If a future
        # model adds another internal fp32 source, retry that forward under
        # torch.autocast("cuda", dtype=torch.bfloat16) instead of patching each site.
        venc.to(dtype=run_dtype)
        model.to(dtype=run_dtype)
        logger.info("Using bf16 compute for DiT and DINOv2 (VAE stays fp32).")

    if args.tile_size <= 0:
        # The DiT only accepts square latents (assert H == W). Tiled inference always
        # feeds square tiles, so the pad/crop shim is only needed on the full-image
        # (non-tiled) path; skip it when --tile_size is active to avoid the overhead.
        model = SquarePadDiT(model, patch_size=getattr(args, 'patch_size', 2))

    # 5. Initialize VOSR
    accelerator = _inference_accelerator_stub()
    vosr_model = VOSR(
        time_dist=args.time_dist,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        interp_type=args.interp_type,
        accelerator=accelerator,
        t_start=args.t_start,
        t_end=args.t_end,
        args=args,
    )
    return vae, venc, model, vosr_model, light_decoder


def main():
    parser = build_parser()

    temp_args, _ = parser.parse_known_args()
    args = load_config_with_cli(temp_args.checkpoint, parser)

    if args.infer_steps is None:
        args.infer_steps = 1
    # vae_tile_size stays None (-> full-image VAE) by default; only explicit
    # --vae_tile_size N enables tiled VAE for extreme (>~8K) inputs.

    # Timestep-embedding structure: whether the student has the auxiliary r_embedder
    # (2nd timestep embedding). Read from the ckpt's args.json — one-step models carry
    # auxiliary_time_cond=False (single t_embedder), distilled students True. Old
    # distilled ckpts lack the key -> default True (backward compatible). With
    # infer_steps=1 the flow-matching recipe is identical for both structures, so the
    # sampling path below needs no branch.
    if getattr(args, 'auxiliary_time_cond', None) is None:
        args.auxiliary_time_cond = True
    args.auxiliary_time_cond = bool(args.auxiliary_time_cond)

    seed = args.seed
    image_paths = list_lq_images(args.input_image)
    if not image_paths:
        logger.warning("No LQ images found. Exit.")
        return

    # Output goes directly into the user-specified -o folder (no tag sub-dir).
    out_dir = args.output_dir

    if (
        not getattr(args, 'force_rerun', False)
        and not os.environ.get('INFER_NO_SKIP', '').lower() in ('1', 'true', 'yes')
        and all_expected_outputs_exist(image_paths, [out_dir])
    ):
        logger.info("Skip: all %d outputs already exist. Use --force_rerun to override.", len(image_paths))
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model from %s...", args.checkpoint)
    logger.info("Inference steps: %d, Seed: %d", args.infer_steps, seed)

    vae, venc, model, vosr_model, light_decoder = load_models(args, device)
    sample_fn = vosr_model.sample_onestep

    # 6. Inference Loop
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir
    to_tensor = transforms.ToTensor()
    logger.info("Processing %d images...", len(image_paths))

    for idx, img_path in enumerate(tqdm(image_paths, file=sys.stderr)):
        img_name = os.path.basename(img_path)
        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size
        target_w, target_h = int(round(w * args.upscale)), int(round(h * args.upscale))
        if idx == 0:
            logger.info("Output size: %dx%d, input size: %dx%d", target_w, target_h, w, h)
            _print_tiling_advice(args, target_w, target_h)

        input_img = raw_img.resize((target_w, target_h), Image.BICUBIC)
        lq = to_tensor(input_img).unsqueeze(0).to(device) * 2.0 - 1.0

        try:
            if args.tile_size > 0:
                sr_tensor = tiled_latent_inference(
                    model, vae, venc, lq, args,
                    device=device, light_decoder=light_decoder,
                )
            else:
                with torch.no_grad():
                    lq_latent, latents_mean, latents_std = encode_dispatch(vae, lq, args, device)
                    z_fea = get_venc_features(venc, lq, args)
                    mdt = next(model.parameters()).dtype
                    lq_latent = lq_latent.to(mdt)
                    z_fea = [f.to(mdt) for f in z_fea]
                    sr_latent = sample_fn(model, lq_latent, n_steps=args.infer_steps, venc_fea=z_fea)
                    sr_tensor = decode_dispatch(vae, sr_latent, args, latents_mean, latents_std, light_decoder)

            sr_img = transforms.ToPILImage()(sr_tensor[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                sr_img = adain_color_fix(sr_img, input_img)
            elif args.align_method == 'wavelet':
                sr_img = wavelet_color_fix(sr_img, input_img)
            sr_img.save(os.path.join(out_dir, os.path.splitext(img_name)[0] + f'_{args.upscale}_{args.tile_size}_{args.vae_tile_size}_{args.dtype}.png'))

        except Exception as e:
            logger.error("Error processing %s: %s", img_name, e)
            import traceback
            traceback.print_exc()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.info("Done!")


if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=3 python inference_vosr_onestep.py \
    -c preset/ckpts/VOSR2 \
    -i preset/datasets/720p \
    -o preset/results24 \
    -u 2 --dtype bf16
 


# VOSR2-1.4B one-step
CUDA_VISIBLE_DEVICES=2 python inference_vosr_onestep.py \
    -c preset/ckpts/VOSR2 \
    -i preset/datasets/inp_data \
    -o preset/results24 \
    -u 2 --force_rerun

# VOSR-0.5B one-step
CUDA_VISIBLE_DEVICES=2 python inference_vosr_onestep.py \
    -c preset/ckpts/VOSR_0.5B_os \
    -i preset/datasets/inp_data \
    -o preset/results24 \
    -u 4

# VOSR-1.4B one-step
CUDA_VISIBLE_DEVICES=2 python inference_vosr_onestep.py \
    -c preset/ckpts/VOSR_1.4B_os \
    -i preset/datasets/inp_data \
    -o preset/results \
    -u 4

# VOSR-1.4B one-step on a large image with DiT + VAE tiling
CUDA_VISIBLE_DEVICES=2 python inference_vosr_onestep.py \
    -c preset/ckpts/VOSR_1.4B_os \
    -i large_input.png \
    -o preset/results \
    -u 4 --tile_size 512 --vae_tile_size 1024  
'''

'''
# Inputs under preset/datasets/inp_data follow RealSR-style evaluation; use --cfg_scale -0.5 (see benchmark presets above).

# VOSR-0.5B multi-step (25 steps)
CUDA_VISIBLE_DEVICES=2 python inference_vosr.py \
    -c preset/ckpts/VOSR_0.5B_ms \
    -i preset/datasets/inp_data \
    -o preset/results24 \
    -u 4

# VOSR-1.4B multi-step (25 steps)
CUDA_VISIBLE_DEVICES=2 python inference_vosr.py \
    -c preset/ckpts/VOSR_1.4B_ms \
    -i preset/datasets/inp_data \
    -o preset/results \
    -u 4
'''
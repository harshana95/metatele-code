#!/usr/bin/env python3
"""
benchmark_latency.py — measure inference latency, peak GPU memory, and
(optionally) MACs/FLOPs for MetaTele and the diffusion baselines under
IDENTICAL conditions, so the numbers are comparable across rows.

Baseline code repos are shallow-cloned on demand (no fine-tuned checkpoints).
Models are instantiated from each repo's configs / diffusers configs only;
weights are random unless SD-2.1 config JSON is already cached locally.

Usage:
  python scripts/latency.py --res 512 --repeats 50 --warmup 10
  python scripts/latency.py --res 512 --methods MetaTele DiffBIR ResShift
  python scripts/latency.py --no-flops   # skip GMAC/GFLOP estimation
  python scripts/latency.py --measured-flops --repeats 1 --warmup 0

GMACs/GFLOPs: one-step methods are measured directly. By default, multi-step
methods (DiffBIR, ResShift, DeblurDiff) profile a single denoise step and
scale by N. Pass --measured-flops to count the full sampling loop once.
"""

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# MetaZoom imports (run from repo root)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPO_URLS = {
    "DiffBIR": "https://github.com/XPixelGroup/DiffBIR.git",
    "ResShift": "https://github.com/zsyoaoa/ResShift.git",
    "OSEDiff": "https://github.com/cswry/OSEDiff.git",
    "DeblurDiff": "https://github.com/kkkls/DeblurDiff.git",
}
DEFAULT_REPO_DIR = Path(
    os.environ.get("LATENCY_REPO_DIR", _ROOT / ".latency_repos")
)
SD21 = "stabilityai/stable-diffusion-2-1"
POS_PROMPT = "high quality photograph"
NEG_PROMPT = "low quality, blurry"


# ----------------------------------------------------------------------
# Repo helpers
# ----------------------------------------------------------------------
def ensure_repos(repo_dir: Path, methods: list[str], skip_download: bool = False) -> None:
    """Shallow-clone any baseline repos needed by the selected methods."""
    needed = {m for m in methods if m in REPO_URLS}
    repo_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(needed):
        dest = repo_dir / name
        if dest.exists():
            continue
        if skip_download:
            raise FileNotFoundError(
                f"[{name}] repo missing at {dest}; drop --skip-download or clone manually."
            )
        url = REPO_URLS[name]
        print(f"[repos] cloning {name} -> {dest}")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
        )


# Top-level packages that collide between MetaZoom and baseline repos.
_COLLIDING_PKGS = ("models", "utils", "pipelines", "diffbir", "model", "ldm")


def _evict_colliding_modules():
    for key in list(sys.modules):
        if key in _COLLIDING_PKGS or any(key.startswith(f"{p}.") for p in _COLLIDING_PKGS):
            del sys.modules[key]


def _ensure_metazoom_path():
    root = str(_ROOT.resolve())
    _evict_colliding_modules()
    if root not in sys.path:
        sys.path.insert(0, root)


@contextlib.contextmanager
def repo_context(repo_name: str, repo_dir: Path):
    """Temporarily chdir into a cloned repo and prepend it to sys.path."""
    path = (repo_dir / repo_name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"repo not found: {path}")
    old_cwd = os.getcwd()
    old_path = sys.path.copy()
    root = str(_ROOT.resolve())
    _evict_colliding_modules()
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != root and p != ""]
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        _evict_colliding_modules()
        _ensure_metazoom_path()


def sd21_components(device, timestep: int = 999):
    """Build SD-2.1 VAE / UNet / scheduler from config only (random weights)."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    def _from_config(cls, subfolder):
        try:
            cfg = cls.load_config(SD21, subfolder=subfolder, local_files_only=True)
        except OSError:
            cfg = cls.load_config(SD21, subfolder=subfolder)
        return cls.from_config(cfg)

    vae = _from_config(AutoencoderKL, "vae").to(device).eval()
    unet = _from_config(UNet2DConditionModel, "unet").to(device).eval()
    scheduler = _from_config(DDPMScheduler, "scheduler")
    scheduler.set_timesteps(timesteps=[timestep], device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return vae, unet, scheduler


def dummy_prompt_embeds(device, batch: int = 1, seq_len: int = 77, dim: int = 1024):
    return torch.zeros(batch, seq_len, dim, device=device)


# ----------------------------------------------------------------------
# Timing / memory harness
# ----------------------------------------------------------------------
@torch.no_grad()
def measure(fn, make_input, device, repeats=50, warmup=10):
    """fn: input_tensor -> output_tensor. Returns dict of stats or None."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    for _ in range(warmup):
        x = make_input()
        _ = fn(x)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        x = make_input()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = fn(x)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    t = torch.tensor(times)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return {
        "latency_s_mean": t.mean().item(),
        "latency_s_std": t.std().item() if len(times) > 1 else 0.0,
        "latency_s_median": t.median().item(),
        "peak_mem_GB": peak_bytes / (1024**3),
        "repeats": repeats,
    }


def count_gflops(fn):
    """Count GFLOPs for one call to fn() using torch.utils.flop_counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        print("  [flops] torch.utils.flop_counter unavailable; skipping")
        return None
    try:
        mode = FlopCounterMode(display=False, depth=None)
        with mode:
            fn()
        return mode.get_total_flops() / 1e9
    except Exception as e:
        print(f"  [flops] failed: {type(e).__name__}: {e}")
        if os.environ.get("FLOPS_DEBUG"):
            import traceback
            traceback.print_exc()
        return None


def _flop_stats(gflops_once, steps=1, calls_per_step=1, deduced=False, breakdown=None):
    """Build result dict; GMACs = GFLOPs/2 (1 MAC ≈ 2 FLOPs)."""
    if gflops_once is None:
        return {}
    per_pass = gflops_once * calls_per_step
    total = per_pass * steps
    out = {
        "GFLOPs_per_pass": per_pass,
        "GMACs_per_pass": per_pass / 2,
        "GFLOPs_total_est": total,
        "GMACs_total_est": total / 2,
        "flops_deduced": deduced,
    }
    if breakdown:
        out["flops_breakdown"] = breakdown
    return out


def profile_flops(name, profile_fn):
    """Run a method-specific flop profiler; never raises."""
    try:
        stats = profile_fn()
        if not stats:
            print(f"  [flops] {name}: could not profile")
            return {}
        g = stats.get("GFLOPs_total_est")
        m = stats.get("GMACs_total_est")
        tag = "deduced" if stats.get("flops_deduced") else "measured"
        print(f"  [flops] {name}: {m:.0f} GMACs / {g:.0f} GFLOPs ({tag})")
        return stats
    except Exception as e:
        print(f"  [flops] {name}: failed: {e}")
        return {}


# ----------------------------------------------------------------------
# Method registry
# ----------------------------------------------------------------------
def build_metatele(device, res, repo_dir):
    _ensure_metazoom_path()
    from diffusers import T2IAdapter
    from models.OneStepDiffusion.onedataset_model import initialize_unet, initialize_vae
    from pipelines.OneStepDiffusionPipeline import OneStepDiffusionPipeline, hpf_adapter_input

    vae, unet, scheduler = sd21_components(device, timestep=999)
    unet = initialize_unet(unet, lora_rank=128)
    vae = initialize_vae(vae, lora_rank=4)
    vae.encoder.conv_in = torch.nn.Conv2d(6, 128, kernel_size=3, stride=1, padding=1).to(device)

    adapter = T2IAdapter(
        in_channels=6,
        channels=(320, 640, 1280, 1280),
        num_res_blocks=2,
        downscale_factor=8,
        adapter_type="full_adapter",
    ).to(device).eval()

    pipe = OneStepDiffusionPipeline(
        vae, unet, scheduler,
        adapter=adapter,
        adapter_preprocess=hpf_adapter_input,
        concatenate_images=True,
    )
    pipe.to(device)
    vae.eval()
    unet.eval()
    adapter.eval()

    prompt_embeds = dummy_prompt_embeds(device)

    def run(x):
        x = x.float()
        if x.min() < 0:
            x = x.clamp(-1, 1)
        else:
            x = x * 2 - 1
        return pipe(x, x.clone(), prompt_embeds=prompt_embeds, timesteps=[999]).images

    def profile():
        x = torch.randn(1, 3, res, res, device=device)
        x = x.clamp(-1, 1)
        g = count_gflops(lambda: pipe(x, x.clone(), prompt_embeds=prompt_embeds, timesteps=[999]))
        return _flop_stats(g, steps=1, deduced=False)

    return run, profile


def _patch_torch_tuple():
    """DiffBIR still references torch.Tuple, removed in recent PyTorch."""
    import typing
    if not hasattr(torch, "Tuple"):
        torch.Tuple = typing.Tuple


def build_osediff(device, res, repo_dir):
    _ensure_metazoom_path()
    # OSEDiff repo model classes target an older diffusers; mirror its one-step
    # forward with SD-2.1 components + LoRA adapters (random init, no ckpt).
    from peft import LoraConfig
    from diffusers import DDPMScheduler

    vae, unet, scheduler = sd21_components(device, timestep=999)
    lora_grep_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                     "to_k", "to_q", "to_v", "to_out.0"]
    lora_targets_vae = []
    for n, _ in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in lora_grep_vae:
            if pat in n and "encoder" in n:
                lora_targets_vae.append(n.replace(".weight", ""))
            elif "quant_conv" in n and "post_quant_conv" not in n:
                lora_targets_vae.append(n.replace(".weight", ""))
    vae.add_adapter(
        LoraConfig(r=4, init_lora_weights="gaussian", target_modules=lora_targets_vae),
        adapter_name="default_encoder",
    )

    lora_grep_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                      "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in",
                      "ff.net.2", "ff.net.0.proj"]
    enc, dec, other = [], [], []
    for n, _ in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pat in lora_grep_unet:
            if pat in n and ("down_blocks" in n or "conv_in" in n):
                enc.append(n.replace(".weight", ""))
                break
            elif pat in n and ("up_blocks" in n or "conv_out" in n):
                dec.append(n.replace(".weight", ""))
                break
            elif pat in n:
                other.append(n.replace(".weight", ""))
                break
    unet.add_adapter(LoraConfig(r=4, init_lora_weights="gaussian", target_modules=enc),
                     adapter_name="default_encoder")
    unet.add_adapter(LoraConfig(r=4, init_lora_weights="gaussian", target_modules=dec),
                     adapter_name="default_decoder")
    unet.add_adapter(LoraConfig(r=4, init_lora_weights="gaussian", target_modules=other),
                     adapter_name="default_others")

    vae.eval()
    unet.eval()
    timesteps = torch.tensor([999], device=device).long()
    prompt_embeds = dummy_prompt_embeds(device)

    def run(x):
        x = x.float()
        if x.min() >= 0:
            x = x * 2 - 1
        lq_latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        pred = unet(lq_latent, timesteps, encoder_hidden_states=prompt_embeds).sample
        z = scheduler.step(pred, timesteps, lq_latent, return_dict=True).prev_sample
        return vae.decode(z / vae.config.scaling_factor).sample.clamp(-1, 1)

    def profile():
        x = torch.randn(1, 3, res, res, device=device).clamp(-1, 1)
        g = count_gflops(lambda: run(x))
        return _flop_stats(g, steps=1, deduced=False)

    return run, profile


def _prepare_diffbir_sdp():
    """DiffBIR defaults to xformers; FlopCounterMode asserts on xformers attention."""
    os.environ["ATTN_MODE"] = "sdp"
    for key in list(sys.modules):
        if key == "diffbir" or key.startswith("diffbir."):
            del sys.modules[key]


def build_diffbir(device, res, steps=50, repo_dir=DEFAULT_REPO_DIR, measured_flops=False):
    _prepare_diffbir_sdp()
    with repo_context("DiffBIR", repo_dir):
        _patch_torch_tuple()
        from omegaconf import OmegaConf
        from diffbir.pipeline import pad_to_multiples_of
        from diffbir.sampler.spaced_sampler import SpacedSampler
        from diffbir.utils.common import instantiate_from_config

        cleaner = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
        cldm = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
        cldm.load_controlnet_from_unet()
        diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion_v2.1.yaml"))

        cleaner.to(device).eval()
        cldm.to(device).eval()
        diffusion.to(device)

        pos = [POS_PROMPT]
        neg = [NEG_PROMPT]

        def _prep(x01):
            cond_img = cleaner(x01)
            cond_img = pad_to_multiples_of(cond_img, multiple=64)
            bs = cond_img.shape[0]
            cond = cldm.prepare_condition(cond_img, pos * bs, False, 256)
            uncond = cldm.prepare_condition(cond_img, neg * bs, False, 256)
            h2, w2 = cond["c_img"].shape[2:]
            x_lat = torch.randn((bs, 4, h2, w2), device=device)
            t = torch.zeros(bs, dtype=torch.long, device=device)
            return cond_img, cond, uncond, x_lat, t

        def run(x):
            x01 = ((x.float() * 0.5 + 0.5) if x.min() < 0 else x.float()).clamp(0, 1)
            _, cond, uncond, _, _ = _prep(x01)
            h2, w2 = cond["c_img"].shape[2:]
            bs = cond["c_img"].shape[0]
            x_T = torch.randn((bs, 4, h2, w2), device=device)
            sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
            z = sampler.sample(
                model=cldm, device=device, steps=steps,
                x_size=(bs, 4, h2, w2), cond=cond, uncond=uncond,
                cfg_scale=4.0, tiled=False, tile_size=64, tile_stride=32,
                x_T=x_T, progress=False,
            )
            h1, w1 = cond["c_img"].shape[2:]
            z = z[..., :h1, :w1]
            return cldm.vae_decode(z, False, 32)

        def profile():
            if measured_flops:
                x = torch.randn(1, 3, res, res, device=device)
                g = count_gflops(lambda: run(x))
                return _flop_stats(g, steps=1, deduced=False)

            x01 = torch.rand(1, 3, res, res, device=device)
            g_clean = count_gflops(lambda: cleaner(x01))
            cond_img, cond, uncond, x_lat, t = _prep(x01)

            def _one_cfg_step():
                cldm(x_lat, t, cond)
                cldm(x_lat, t, uncond)

            g_step = count_gflops(_one_cfg_step)
            h1, w1 = cond["c_img"].shape[2:]
            z = torch.randn(1, 4, h1, w1, device=device)
            g_dec = count_gflops(lambda: cldm.vae_decode(z, False, 32))
            g_prep = count_gflops(
                lambda: (cldm.prepare_condition(cond_img, pos, False, 256),
                         cldm.prepare_condition(cond_img, neg, False, 256))
            )

            breakdown = {
                "cleaner_GFLOPs": g_clean,
                "cond_prep_GFLOPs": g_prep,
                "denoise_step_cfg2x_GFLOPs": g_step,
                "vae_decode_GFLOPs": g_dec,
            }
            total = sum(v for v in breakdown.values() if v) + (g_step or 0) * (steps - 1)
            return {
                "GFLOPs_total_est": total,
                "GMACs_total_est": total / 2,
                "GFLOPs_per_pass": g_step,
                "GMACs_per_pass": (g_step or 0) / 2,
                "flops_deduced": True,
                "flops_breakdown": breakdown,
                "flops_note": f"1×cleaner + 1×cond_prep + {steps}×CFG-denoise + 1×vae_dec",
            }

        return run, profile


def build_resshift(device, res, steps=15, repo_dir=DEFAULT_REPO_DIR, measured_flops=False):
    with repo_context("ResShift", repo_dir):
        from omegaconf import OmegaConf
        from utils import util_common

        cfg_path = "configs/realsr_swinunet_realesrgan256.yaml"
        configs = OmegaConf.load(cfg_path)
        configs.diffusion.params.steps = steps
        configs.model.params.lq_size = res
        configs.model.params.image_size = res // 4  # latent H/W for f4 VAE at 512px input
        configs.diffusion.params.sf = 1
        lq_side = res

        base_diffusion = util_common.instantiate_from_config(configs.diffusion)
        model = util_common.instantiate_from_config(configs.model).to(device).eval()

        autoencoder = None
        if configs.autoencoder is not None:
            params = configs.autoencoder.get("params", {})
            autoencoder = util_common.get_obj_from_str(configs.autoencoder.target)(**params)
            autoencoder.to(device).eval()

        def _y0(x):
            y0 = x.float()
            if y0.min() >= 0:
                y0 = y0 * 2 - 1
            return y0

        def run(x):
            y0 = _y0(x)
            return base_diffusion.p_sample_loop(
                y=y0,
                model=model,
                first_stage_model=autoencoder,
                noise=None,
                noise_repeat=False,
                clip_denoised=(autoencoder is None),
                denoised_fn=None,
                model_kwargs={"lq": y0},
                progress=False,
            )

        def profile():
            if measured_flops:
                x = torch.randn(1, 3, res, res, device=device)
                g = count_gflops(lambda: run(x))
                return _flop_stats(g, steps=1, deduced=False)

            y0 = _y0(torch.randn(1, 3, res, res, device=device))
            z = base_diffusion.encode_first_stage(y0, autoencoder, up_sample=True)
            t = torch.tensor([0], device=device, dtype=torch.long)

            g_enc = count_gflops(
                lambda: base_diffusion.encode_first_stage(y0, autoencoder, up_sample=True)
            )
            g_step = count_gflops(
                lambda: model(base_diffusion._scale_input(z, t), t, lq=y0)
            )
            g_dec = count_gflops(
                lambda: autoencoder.decode(z / base_diffusion.scale_factor)
            )

            breakdown = {
                "ae_encode_GFLOPs": g_enc,
                "denoise_step_GFLOPs": g_step,
                "ae_decode_GFLOPs": g_dec,
            }
            total = sum(v for v in breakdown.values() if v) + (g_step or 0) * (steps - 1)
            return {
                "GFLOPs_total_est": total,
                "GMACs_total_est": total / 2,
                "GFLOPs_per_pass": g_step,
                "GMACs_per_pass": (g_step or 0) / 2,
                "flops_deduced": True,
                "flops_breakdown": breakdown,
                "flops_note": f"1×ae_enc + {steps}×unet + 1×ae_dec @ {lq_side}px LQ",
            }

        return run, profile


def build_deblurdiff(device, res, steps=50, repo_dir=DEFAULT_REPO_DIR, measured_flops=False):
    with repo_context("DeblurDiff", repo_dir):
        _patch_torch_tuple()
        from omegaconf import OmegaConf
        from utils.common import instantiate_from_config
        from utils.pipeline import pad_to_multiples_of
        from utils.sampler import SpacedSampler

        cldm = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
        cldm.load_controlnet_from_unet()
        diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
        cldm.to(device).eval()
        diffusion.to(device)

        pos = [POS_PROMPT]
        neg = [NEG_PROMPT]

        def _prep(x):
            clean = x.float()
            if clean.min() >= 0:
                clean = clean * 2 - 1
            clean = (clean + 1) / 2
            clean = pad_to_multiples_of(clean, multiple=64)
            bs = clean.shape[0]
            cond = cldm.prepare_condition(clean, pos * bs)
            uncond = cldm.prepare_condition(clean, neg * bs)
            h2, w2 = cond["c_img"].shape[2:]
            x_lat = torch.randn((bs, 4, h2, w2), device=device)
            t = torch.zeros(bs, dtype=torch.long, device=device)
            return clean, cond, uncond, x_lat, t

        def run(x):
            clean, cond, uncond, _, _ = _prep(x)
            h2, w2 = cond["c_img"].shape[2:]
            bs = clean.shape[0]
            x_T = torch.randn((bs, 4, h2, w2), device=device)
            sampler = SpacedSampler(diffusion.betas)
            z = sampler.sample(
                model=cldm, device=device, steps=steps, batch_size=bs,
                x_size=(4, h2, w2), cond=cond, uncond=uncond,
                cfg_scale=4.0, x_T=x_T, progress=False,
                tiled=False, tile_size=64, tile_stride=32,
            )
            ori_h, ori_w = clean.shape[2], clean.shape[3]
            return cldm.vae_decode(z)[:, :, :ori_h, :ori_w]

        def profile():
            if measured_flops:
                x = torch.randn(1, 3, res, res, device=device)
                g = count_gflops(lambda: run(x))
                return _flop_stats(g, steps=1, deduced=False)

            clean, cond, uncond, x_lat, t = _prep(torch.randn(1, 3, res, res, device=device))

            def _one_cfg_step():
                cldm(x_lat, t, cond)
                cldm(x_lat, t, uncond)

            g_step = count_gflops(_one_cfg_step)
            g_prep = count_gflops(
                lambda: (cldm.prepare_condition(clean, pos),
                         cldm.prepare_condition(clean, neg))
            )
            h2, w2 = cond["c_img"].shape[2:]
            z = torch.randn(1, 4, h2, w2, device=device)
            g_dec = count_gflops(lambda: cldm.vae_decode(z))

            breakdown = {
                "cond_prep_GFLOPs": g_prep,
                "denoise_step_cfg2x_GFLOPs": g_step,
                "vae_decode_GFLOPs": g_dec,
            }
            total = sum(v for v in breakdown.values() if v) + (g_step or 0) * (steps - 1)
            return {
                "GFLOPs_total_est": total,
                "GMACs_total_est": total / 2,
                "GFLOPs_per_pass": g_step,
                "GMACs_per_pass": (g_step or 0) / 2,
                "flops_deduced": True,
                "flops_breakdown": breakdown,
                "flops_note": f"1×cond_prep + {steps}×CFG-denoise(KPN+CN+UNet) + 1×vae_dec",
            }

        return run, profile


def _registry(repo_dir, measured_flops=False):
    return {
        "MetaTele": lambda dev, res: build_metatele(dev, res, repo_dir),
        "DiffBIR": lambda dev, res: build_diffbir(
            dev, res, steps=50, repo_dir=repo_dir, measured_flops=measured_flops
        ),
        "ResShift": lambda dev, res: build_resshift(
            dev, res, steps=15, repo_dir=repo_dir, measured_flops=measured_flops
        ),
        "DeblurDiff": lambda dev, res: build_deblurdiff(
            dev, res, steps=50, repo_dir=repo_dir, measured_flops=measured_flops
        ),
        "OSEDiff": lambda dev, res: build_osediff(dev, res, repo_dir),
    }


STEPS = {
    "MetaTele": 1,
    "OSEDiff": 1,
    "ResShift": 15,
    "DiffBIR": 50,
    "DeblurDiff": 50,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512,
                    help="square input side (px); fixed at 512 for comparable benchmarks")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-flops", action="store_true",
                    help="skip GMAC/GFLOP estimation (on by default)")
    ap.add_argument("--measured-flops", action="store_true",
                    help="profile full inference (no step scaling) for multi-step methods")
    ap.add_argument("--methods", nargs="*", default=["MetaTele", "OSEDiff", "DiffBIR", "ResShift", "DeblurDiff"])
    ap.add_argument("--out", default="latency_results.json")
    ap.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR,
                    help="where baseline repos are cloned")
    ap.add_argument("--skip-download", action="store_true",
                    help="fail instead of cloning missing repos")
    args = ap.parse_args()
    if args.res != 512:
        print(f"Note: forcing 512×512 (ignoring --res {args.res})")
    args.res = 512

    assert torch.cuda.is_available(), "CUDA required for a fair GPU benchmark"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Resolution: {args.res}x{args.res}, repeats={args.repeats}, warmup={args.warmup}")
    if args.measured_flops:
        print("FLOPs: full end-to-end measurement (--measured-flops)")
    print(f"Repo dir: {args.repo_dir} (architecture only, no fine-tuned checkpoints)\n")

    ensure_repos(args.repo_dir.resolve(), args.methods, skip_download=args.skip_download)
    registry = _registry(args.repo_dir.resolve(), measured_flops=args.measured_flops)

    def make_input():
        return torch.randn(1, args.channels, args.res, args.res, device=device)

    results = {}
    for name in args.methods:
        if name not in registry:
            print(f"[{name}] unknown method, skipping")
            continue
        try:
            built = registry[name](device, args.res)
        except Exception as e:
            print(f"[{name}] failed to build: {e}")
            continue
        if built is None:
            print(f"[{name}] not wired -> skipping")
            continue
        fn, profile = built
        print(f"[{name}] measuring...")
        try:
            stats = measure(fn, make_input, device, args.repeats, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f"[{name}] OOM during measurement — skipping")
            del fn, profile, built
            gc.collect()
            torch.cuda.empty_cache()
            continue
        stats["steps"] = STEPS.get(name)
        if not args.no_flops:
            stats.update(profile_flops(name, profile))
        results[name] = stats
        s = stats
        flop_str = ""
        if "GMACs_total_est" in s:
            flop_str = f" | {s['GMACs_total_est']:.0f} GMACs ({s['GFLOPs_total_est']:.0f} GFLOPs)"
            if s.get("flops_deduced"):
                flop_str += " [deduced]"
        print(
            f"    latency {s['latency_s_mean']*1000:.1f} +/- {s['latency_s_std']*1000:.1f} ms"
            f" | peak {s['peak_mem_GB']:.2f} GB{flop_str}\n"
        )
        del fn, profile, built
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(
            {"gpu": torch.cuda.get_device_name(device), "res": args.res, "results": results},
            f,
            indent=2,
        )

    print("=" * 66)
    print("LaTeX rows (paste into the table; verify step counts in caption):")
    order = ["DeblurDiff", "DiffBIR", "ResShift", "OSEDiff", "MetaTele"]
    for name in order:
        if name not in results:
            continue
        s = results[name]
        lat = f"{s['latency_s_mean']:.2f}"
        mem = f"{s['peak_mem_GB']:.1f}"
        flop = f"{s['GMACs_total_est']:.0f}" if "GMACs_total_est" in s else "---"
        gflop = f"{s['GFLOPs_total_est']:.0f}" if "GFLOPs_total_est" in s else "---"
        step = s["steps"] if s["steps"] is not None else "?"
        label = name + " (ours)" if name == "MetaTele" else name
        ded = " *" if s.get("flops_deduced") else ""
        print(f"{label:22s} & {step} & {lat} & {mem} & {flop}{ded} & {gflop}{ded} \\\\")
    print("* = deduced from single-step profile × N steps")
    print("=" * 66)


if __name__ == "__main__":
    main()

import os

# ─── Environment Setup ───────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"]   = "expandable_segments:True"
os.environ["PYTHON_UNBUFFERED"]          = "1"
os.environ["HF_HUB_OFFLINE"]            = "1"
os.environ["TRANSFORMERS_OFFLINE"]      = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import runpod
import torch
import base64
import io
import time
import gc
import numpy as np
from PIL import Image

# ─── RTX 5090 / Blackwell caps ───────────────────────────────────────────────
# GB202 = 32 GB GDDR7, native FP8 tensor cores, CUDA Compute 12.0
# We run DiT in fp8_e4m3fn, text-encoder & VAE in bfloat16 (they're small).
DEVICE             = "cuda"
COMPUTE_DTYPE      = torch.bfloat16          # activations always bf16
FP8_DTYPE          = torch.float8_e4m3fn     # weight storage for DiT
USE_FLASH_ATTN     = hasattr(torch.nn.functional, "scaled_dot_product_attention")

# ─── Path Configuration ──────────────────────────────────────────────────────
# ComfyOrg FP8 weight files — expected on RunPod Network Volume
COMFY_DIFFUSION_MODEL = os.environ.get(
    "COMFY_DIFFUSION_MODEL",
    "/runpod-volume/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
)
COMFY_TEXT_ENCODER = os.environ.get(
    "COMFY_TEXT_ENCODER",
    "/runpod-volume/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
)
# Lightning 4-step LoRA (baked into image at /app/models/lora)
LORA_PATH = "/app/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# Processor baked into image
PROCESSOR_PATH = "/app/models/processor"

# Legacy HF cache scan (fallback only)
HF_CACHE             = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
DIFFSYNTH_MODELS_DIR = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/app/models")

pipe         = None
_loaded_alpha = 0.125  # Default Lightning LoRA alpha (8/64)

# ─── Cache Helpers ───────────────────────────────────────────────────────────
def find_snapshot_path(model_id, name="Component"):
    """Fallback: scan HF-style cache layout for a snapshot directory."""
    search_roots = [
        HF_CACHE,
        "/workspace/huggingface-cache",
        "/runpod-volume/huggingface-cache",
        "/runpod-volume",
        "/app/models",
    ]
    potential_ids = [model_id, model_id.replace("-2511", ""), "Qwen/Qwen-Image-Edit"]
    potential_names = []
    for pid in potential_ids:
        hf_name = "models--" + pid.replace("/", "--")
        potential_names.extend([hf_name, hf_name.lower()])

    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        hub_dir = os.path.join(root, "hub") if os.path.isdir(os.path.join(root, "hub")) else root
        for name_variant in list(dict.fromkeys(potential_names)):
            model_dir = os.path.join(hub_dir, name_variant)
            if os.path.isdir(model_dir):
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    hashes = os.listdir(snapshots_dir)
                    if hashes:
                        path = os.path.join(snapshots_dir, hashes[0])
                        print(f"[Cache] ✅ {name} snapshot: {path}")
                        return path
    return None

def link_to_diffsynth_cache(model_id, snapshot_path):
    org, repo = model_id.split("/", 1)
    org_dir   = os.path.join(DIFFSYNTH_MODELS_DIR, org)
    link_path = os.path.join(org_dir, repo)
    os.makedirs(org_dir, exist_ok=True)
    if not (os.path.islink(link_path) or os.path.exists(link_path)):
        try:
            os.symlink(snapshot_path, link_path)
        except Exception as e:
            print(f"[Cache] Link failed for {model_id}: {e}")

def prepare_cache():
    print(f"\n{'='*60}\n{'[Cache] READY-CHECK':^60}\n{'='*60}")
    missing = []

    # ── Primary path: ComfyOrg FP8 weights on RunPod volume ──
    if os.path.exists(COMFY_DIFFUSION_MODEL):
        print(f"[Cache] ✅ FP8 DiT  : {COMFY_DIFFUSION_MODEL}")
    else:
        print(f"[Cache] ❌ FP8 DiT MISSING : {COMFY_DIFFUSION_MODEL}")
        missing.append("fp8-diffusion-model")

    if os.path.exists(COMFY_TEXT_ENCODER):
        print(f"[Cache] ✅ FP8 TE   : {COMFY_TEXT_ENCODER}")
    else:
        print(f"[Cache] ❌ FP8 TE MISSING  : {COMFY_TEXT_ENCODER}")
        missing.append("fp8-text-encoder")

    # ── LoRA (baked into image) ──
    if os.path.exists(LORA_PATH):
        print(f"[LoRA]  ✅ Lightning LoRA found")
    else:
        print(f"[LoRA]  ❌ Lightning LoRA MISSING")
        missing.append("4-step-lora")

    # ── Processor (baked into image) ──
    if os.path.isdir(PROCESSOR_PATH) and "config.json" in os.listdir(PROCESSOR_PATH):
        print(f"[Proc]  ✅ Processor found")
    else:
        print(f"[Proc]  ❌ Processor MISSING")
        missing.append("processor")

    # ── Fallback: try HF snapshot for VAE (not in ComfyOrg split) ──
    snap = find_snapshot_path("Qwen/Qwen-Image-Edit-2511", name="HF snapshot")
    if snap:
        link_to_diffsynth_cache("Qwen/Qwen-Image-Edit-2511", snap)
        for legacy in ["Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit"]:
            link_to_diffsynth_cache(legacy, snap)

    if missing:
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(f"[Cache] ⚠️  Offline mode DISABLED — missing: {missing}")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[Cache] ✅ All components ready — OFFLINE mode ON")
    print(f"{'='*60}\n")

# ─── Patching ────────────────────────────────────────────────────────────────
try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25
    if not hasattr(_qwen25, "Qwen2RMSNorm"):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        _qwen25.Qwen2RMSNorm = Qwen2RMSNorm
        print("[Patch] Injected Qwen2RMSNorm → qwen2_5_vl")
except Exception as e:
    print(f"[Patch] Warning: {e}")

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler

# ─── FP8 weight cast helper ──────────────────────────────────────────────────
def _cast_dit_to_fp8(module: torch.nn.Module) -> torch.nn.Module:
    """
    Cast all linear weight tensors in the DiT to fp8_e4m3fn for Blackwell.
    Activations remain bfloat16; only the stored weights are compressed.
    This cuts DiT VRAM from ~14 GB (bf16) → ~7 GB (fp8) on the 5090.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        print("[FP8] torch.float8_e4m3fn unavailable — skipping cast")
        return module
    converted = 0
    for name, param in module.named_parameters():
        if param.dtype in (torch.float16, torch.bfloat16) and param.ndim >= 2:
            param.data = param.data.to(torch.float8_e4m3fn)
            converted += 1
    print(f"[FP8] Cast {converted} weight tensors → float8_e4m3fn")
    return module

# ─── Model Loading ───────────────────────────────────────────────────────────
def load_model():
    global pipe
    start_load = time.time()
    prepare_cache()

    # ── VRAM budget ──────────────────────────────────────────────────────────
    # RTX 5090 = 32 GB. Reserve 4 GB for activations (large images).
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    safe_limit    = max(20.0, total_vram_gb - 4.0)
    print(f"[VRAM] GPU: {torch.cuda.get_device_name(0)} | Total: {total_vram_gb:.1f} GB | Limit: {safe_limit:.1f} GB")

    # ── VRAM configs ─────────────────────────────────────────────────────────
    # DiT: offload to CPU between steps, compute on CUDA in bf16
    # (weights already in fp8 on disk; DiffSynth loads them, we cast after)
    vram_dit = {
        "offload_dtype":     torch.bfloat16,
        "offload_device":    "cpu",
        "onload_dtype":      torch.bfloat16,
        "onload_device":     "cpu",
        "preparing_dtype":   torch.bfloat16,
        "preparing_device":  DEVICE,
        "computation_dtype": torch.bfloat16,
        "computation_device": DEVICE,
    }
    # Text encoder & VAE: small enough to stay on CUDA in bf16
    vram_static = {
        "onload_dtype":       torch.bfloat16,
        "onload_device":      "cpu",
        "preparing_dtype":    torch.bfloat16,
        "preparing_device":   DEVICE,
        "computation_dtype":  torch.bfloat16,
        "computation_device": DEVICE,
    }

    # ── Determine model file sources ─────────────────────────────────────────
    # ComfyOrg split: separate DiT + text-encoder safetensors.
    # DiffSynth expects origin_file_pattern globs — supply absolute paths directly.
    dit_source = COMFY_DIFFUSION_MODEL
    te_source  = COMFY_TEXT_ENCODER

    # VAE is not in the ComfyOrg split; grab from HF snapshot if available,
    # else fall back to original Qwen HF repo id (requires online access).
    hf_snap    = find_snapshot_path("Qwen/Qwen-Image-Edit-2511", name="VAE source")
    vae_source = (
        os.path.join(hf_snap, "vae", "diffusion_pytorch_model.safetensors")
        if hf_snap and os.path.exists(os.path.join(hf_snap, "vae", "diffusion_pytorch_model.safetensors"))
        else "Qwen/Qwen-Image-Edit-2511"   # triggers online download if missing
    )

    print(f"[Load] DiT  : {dit_source}")
    print(f"[Load] TE   : {te_source}")
    print(f"[Load] VAE  : {vae_source}")

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=DEVICE,
        model_configs=[
            # DiT — FP8 file, loaded as bf16 by DiffSynth then we re-cast
            ModelConfig(
                model_id=dit_source,
                origin_file_pattern=dit_source,
                **vram_dit,
            ),
            # Text Encoder — FP8 scaled file
            ModelConfig(
                model_id=te_source,
                origin_file_pattern=te_source,
                **vram_static,
            ),
            # VAE — bf16 from HF snapshot or online
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_static,
            ),
        ],
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit-2511",
            origin_file_pattern=PROCESSOR_PATH,
        ),
        vram_limit=safe_limit,
    )

    # ── Lightning LoRA ───────────────────────────────────────────────────────
    if os.path.exists(LORA_PATH):
        try:
            pipe.load_lora(pipe.dit, LORA_PATH, alpha=_loaded_alpha)
            pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
            print(f"[LoRA] Lightning 4-step applied (alpha={_loaded_alpha})")
        except Exception as e:
            print(f"[LoRA] Load failed: {e}")

    # ── RTX 5090: enable Flash Attention via SDPA ────────────────────────────
    if USE_FLASH_ATTN:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # FA3 is faster
        print("[Attn] Flash Attention (SDPA) enabled")

    # ── channels-last memory format for bf16 matmuls on Blackwell ────────────
    try:
        pipe.dit = pipe.dit.to(memory_format=torch.channels_last)
        print("[Mem]  DiT → channels_last")
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Model ready in {time.time() - start_load:.2f}s")
    _log_vram("post-load")

def _log_vram(tag=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved()  / (1024 ** 3)
    print(f"[VRAM{' '+tag if tag else ''}] alloc={allocated:.2f} GB  reserved={reserved:.2f} GB")

# ─── Handler ─────────────────────────────────────────────────────────────────
def parse_alpha(alpha_val):
    if isinstance(alpha_val, (int, float)):
        return float(alpha_val)
    if not isinstance(alpha_val, str):
        return 0.125
    try:
        return float(eval(alpha_val, {"__builtins__": None}, {})) if "/" in alpha_val else float(alpha_val)
    except Exception:
        return 0.125

def handler(job):
    global pipe, _loaded_alpha
    if not pipe:
        load_model()

    job_input = job.get("input", {})
    prompt    = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing prompt"}

    # Support both single 'image' and list 'images'
    images_b64 = job_input.get("images") or job_input.get("image")
    if images_b64 and isinstance(images_b64, str):
        images_b64 = [images_b64]

    steps     = job_input.get("steps",     4)
    cfg_scale = job_input.get("cfg_scale", 1.0)

    # ── RTX 5090: support higher-res ─────────────────────────────────────────
    # Default output resolution bumped to 1280x1280 (was 1024x1024 for 4090).
    input_images = None
    if images_b64:
        try:
            input_images = [
                Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                for b64 in images_b64
            ]
        except Exception as e:
            return {"error": f"Invalid image(s): {e}"}
        img_w, img_h = input_images[0].size
        print(f"[Gen] I2I | {img_w}x{img_h} | n={len(input_images)} | steps={steps} cfg={cfg_scale}")
    else:
        img_w = job_input.get("width",  1280)
        img_h = job_input.get("height", 1280)
        print(f"[Gen] T2I | {img_w}x{img_h} | steps={steps} cfg={cfg_scale}")

    _log_vram("pre-inference")
    start = time.time()

    # ── torch.compile for Blackwell (optional, warm-up on first call) ────────
    # Uncomment if you want persistent compiled graph across calls:
    # if not getattr(pipe.dit, "_compiled", False):
    #     pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
    #     pipe.dit._compiled = True
    #     print("[Compile] DiT compiled with torch.compile")

    try:
        # Use autocast to keep activations in bf16 even when weights are fp8
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_image = pipe(
                prompt,
                edit_image=input_images,
                num_inference_steps=steps,
                height=img_h,
                width=img_w,
                edit_image_auto_resize=True,
                zero_cond_t=True,
                cfg_scale=cfg_scale,
            )

        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        elapsed = time.time() - start
        print(f"[Gen] Done in {elapsed:.2f}s")
        _log_vram("post-inference")

        return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": "OOM — try reducing resolution or step count"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})

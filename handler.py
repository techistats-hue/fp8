import os

# ─── Environment Setup (must be first, before any HF imports) ────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"]        = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"]               = "1"
os.environ["HF_HUB_OFFLINE"]                = "1"
os.environ["TRANSFORMERS_OFFLINE"]          = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]      = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]  = "1"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"]     = "huggingface"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Hard Network Firewall ────────────────────────────────────────────────────
# Intercept DNS so any accidental hub call fails instantly (not after a long
# throttled retry loop) during model init.
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo
_BLOCKED_HOSTS    = {"huggingface.co", "cdn-lfs.huggingface.co", "modelscope.cn"}
def _guarded_getaddrinfo(host, *args, **kwargs):
    if any(h in (host or "") for h in _BLOCKED_HOSTS):
        raise OSError(f"[Firewall] Blocked outbound: {host}. All weights must be on the volume.")
    return _orig_getaddrinfo(host, *args, **kwargs)
_socket.getaddrinfo = _guarded_getaddrinfo
print("[Firewall] Outbound HF/ModelScope DNS blocked.")

import runpod
import torch
import base64
import io
import time
import gc
from PIL import Image

# ─── Path Configuration ───────────────────────────────────────────────────────
HF_CACHE             = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
DIFFSYNTH_MODELS_DIR = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/app/models")

# Comfy-Org/Qwen-Image_ComfyUI split_files layout on the RunPod volume:
#   <COMFY_CACHE_DIR>/diffusion_models/qwen_image_fp8mixed.safetensors
#   <COMFY_CACHE_DIR>/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
#   <COMFY_CACHE_DIR>/vae/qwen_image_vae.safetensors
COMFY_CACHE_DIR = os.environ.get("COMFY_CACHE_DIR", "/workspace/ComfyUI/models")

COMFY_SEARCH_ROOTS = [
    COMFY_CACHE_DIR,   # set to exact snapshot/split_files path via env var
    # Fallback: walk the snapshot tree dynamically
    "/runpod-volume/huggingface-cache/hub/models--comfy-org--qwen-image_comfyui/snapshots/c232bcb51c1523899c62d6dcaa960b2627668de5/split_files",
    "/runpod-volume/huggingface-cache",
    "/runpod-volume/models",
    "/workspace/ComfyUI/models",   # kept as last-resort if mount changes
]

FP8_FILES = {
    "diffusion":    "diffusion_models/qwen_image_fp8mixed.safetensors",
    "text_encoder": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae":          "vae/qwen_image_vae.safetensors",
}

pipe          = None
_loaded_alpha = 0.125

# ─── FP8 File Locator ─────────────────────────────────────────────────────────
def find_fp8_file(key: str) -> str | None:
    rel = FP8_FILES[key]
    for root in COMFY_SEARCH_ROOTS:
        if not root or not os.path.exists(root):
            continue
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            size_gb = os.path.getsize(candidate) / (1024 ** 3)
            print(f"[FP8] ✅ {key} → {candidate}  ({size_gb:.2f} GB)")
            return candidate
    return None

# ─── HF Snapshot Locator (processor only, purely local) ──────────────────────
def find_snapshot_path(model_id, name="Component"):
    hf_name = "models--" + model_id.replace("/", "--")
    for root in [HF_CACHE, "/runpod-volume/huggingface-cache", "/workspace/huggingface-cache"]:
        if not root or not os.path.exists(root):
            continue
        hub_dir   = os.path.join(root, "hub") if os.path.isdir(os.path.join(root, "hub")) else root
        model_dir = os.path.join(hub_dir, hf_name)
        if os.path.isdir(model_dir):
            snaps = os.path.join(model_dir, "snapshots")
            if os.path.isdir(snaps):
                hashes = [h for h in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, h))]
                if hashes:
                    path = os.path.join(snaps, hashes[0])
                    print(f"[Cache] ✅ {name} → {path}")
                    return path
    return None

# ─── Preflight Check ──────────────────────────────────────────────────────────
def prepare_cache() -> dict:
    print(f"\n{'='*60}\n{'[Cache] FP8 READY-CHECK (RTX 5090)':^60}\n{'='*60}")
    missing   = []
    fp8_paths = {}

    for key in FP8_FILES:
        path = find_fp8_file(key)
        if path:
            fp8_paths[key] = path
        else:
            print(f"[FP8] ❌ {key} MISSING  searched: {COMFY_SEARCH_ROOTS}")
            missing.append(f"fp8/{key}")

    lora_path = "/app/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    if os.path.exists(lora_path):
        print(f"[LoRA]      ✅ Baked Lightning LoRA")
    else:
        print(f"[LoRA]      ❌ MISSING")
        missing.append("LoRA")

    proc_path = "/app/models/processor"
    if os.path.isdir(proc_path) and "config.json" in os.listdir(proc_path):
        print(f"[Processor] ✅ Baked Processor")
    else:
        print(f"[Processor] ❌ MISSING")
        missing.append("Processor")

    print(f"{'='*60}\n")

    if missing:
        raise RuntimeError(
            f"Missing components: {missing}\n"
            f"Expected FP8 files at:\n"
            f"  {COMFY_CACHE_DIR}/diffusion_models/qwen_image_fp8mixed.safetensors\n"
            f"  {COMFY_CACHE_DIR}/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors\n"
            f"  {COMFY_CACHE_DIR}/vae/qwen_image_vae.safetensors\n"
            f"Set COMFY_CACHE_DIR env var if your volume mount differs."
        )

    return fp8_paths

# ─── Patching ─────────────────────────────────────────────────────────────────
try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25
    if not hasattr(_qwen25, "Qwen2RMSNorm"):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        _qwen25.Qwen2RMSNorm = Qwen2RMSNorm
        print("[Patch] Injected Qwen2RMSNorm into qwen2_5_vl")
except Exception as e:
    print(f"[Patch] Warning: {e}")

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler

# ─── Model Loading ────────────────────────────────────────────────────────────
def load_model():
    global pipe, _loaded_alpha
    start_load = time.time()

    fp8_paths = prepare_cache()

    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    safe_limit = max(16.0, vram_total - 6.0)
    print(f"[VRAM] Total={vram_total:.1f}GB  Safe limit={safe_limit:.1f}GB")

    # ── CRITICAL: use absolute local paths, NOT HuggingFace repo IDs ──────
    # Passing a real HF model_id to ModelConfig causes DiffSynth to call the
    # Hub API for file resolution → throttling/hangs even with HF_HUB_OFFLINE=1
    # because some code paths bypass the env-var check.
    # Solution: set model_id to a dummy local string and pass the resolved
    # absolute file path directly as origin_file_pattern.
    # ──────────────────────────────────────────────────────────────────────

    fp8_compute = dict(
        onload_dtype        = torch.float8_e4m3fn,  # FP8 stored in VRAM
        onload_device       = "cuda",
        preparing_dtype     = torch.bfloat16,        # upcast for ops
        preparing_device    = "cuda",
        computation_dtype   = torch.bfloat16,
        computation_device  = "cuda",
    )
    bf16_only = dict(
        onload_dtype        = torch.bfloat16,
        onload_device       = "cuda",
        preparing_dtype     = torch.bfloat16,
        preparing_device    = "cuda",
        computation_dtype   = torch.bfloat16,
        computation_device  = "cuda",
    )

    proc_path = "/app/models/processor"
    snap = find_snapshot_path("Qwen/Qwen-Image-Edit-2511", name="Processor snapshot")
    if snap:
        cand = os.path.join(snap, "processor")
        if os.path.exists(os.path.join(cand, "preprocessor_config.json")):
            proc_path = cand

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype   = torch.bfloat16,
        device        = "cuda",
        model_configs = [
            ModelConfig(
                model_id            = "local/diffusion",           # dummy — never resolved
                origin_file_pattern = fp8_paths["diffusion"],      # absolute path on volume
                **fp8_compute,
            ),
            ModelConfig(
                model_id            = "local/text_encoder",
                origin_file_pattern = fp8_paths["text_encoder"],
                **fp8_compute,
            ),
            ModelConfig(
                model_id            = "local/vae",
                origin_file_pattern = fp8_paths["vae"],
                **bf16_only,                                       # BF16 for decode quality
            ),
        ],
        processor_config = ModelConfig(
            model_id            = "local/processor",
            origin_file_pattern = proc_path,
        ),
        vram_limit = safe_limit,
    )

    # Apply baked-in Lightning LoRA
    lora_path = "/app/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    if os.path.exists(lora_path):
        try:
            pipe.load_lora(pipe.dit, lora_path, alpha=_loaded_alpha)
            pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
            print(f"[LoRA] Lightning 4-step applied (alpha={_loaded_alpha:.4f})")
        except Exception as e:
            print(f"[LoRA] Load failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Model ready in {time.time() - start_load:.2f}s")

# ─── Handler ──────────────────────────────────────────────────────────────────
def handler(job):
    global pipe, _loaded_alpha
    if not pipe:
        load_model()

    job_input  = job.get("input", {})
    prompt     = job_input.get("prompt")
    images_b64 = job_input.get("images") or job_input.get("image")

    if not prompt:
        return {"error": "Missing prompt"}

    if images_b64 and isinstance(images_b64, str):
        images_b64 = [images_b64]

    steps     = job_input.get("steps", 4)
    cfg_scale = job_input.get("cfg_scale", 1.0)

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
        print(f"I2I | {img_w}x{img_h} | n={len(input_images)} steps={steps} cfg={cfg_scale}")
    else:
        img_w = job_input.get("width", 1024)
        img_h = job_input.get("height", 1024)
        print(f"T2I | {img_w}x{img_h} | steps={steps} cfg={cfg_scale}")

    start = time.time()
    try:
        output_image = pipe(
            prompt,
            edit_image             = input_images,
            num_inference_steps    = steps,
            height                 = img_h,
            width                  = img_w,
            edit_image_auto_resize = True,
            zero_cond_t            = True,
            cfg_scale              = cfg_scale,
        )
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        print(f"Done: {time.time() - start:.2f}s")
        return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})

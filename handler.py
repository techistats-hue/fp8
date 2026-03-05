import os

# ─── Environment Setup ───────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHON_UNBUFFERED"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import runpod
import torch
import base64
import io
import time
import gc
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Cache & Path Configuration ──────────────────────────────────────────────
HF_CACHE             = os.environ.get("HF_HOME", "/workspace/huggingface-cache")
DIFFSYNTH_MODELS_DIR = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/app/models")

# Comfy-Org split_files layout on the RunPod network volume.
# Expected tree under COMFY_CACHE_DIR:
#   diffusion_models/qwen_image_fp8mixed.safetensors
#   text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
#   vae/qwen_image_vae.safetensors
COMFY_CACHE_DIR = os.environ.get(
    "COMFY_CACHE_DIR",
    "/workspace/ComfyUI/models"   # default RunPod ComfyUI volume path
)

# Candidate roots to search for the Comfy split-file layout.
# Add more as needed for non-standard RunPod mount points.
COMFY_SEARCH_ROOTS = [
    COMFY_CACHE_DIR,
    "/workspace/ComfyUI/models",
    "/runpod-volume/ComfyUI/models",
    "/runpod-volume/models",
]

pipe = None
_loaded_alpha = 0.125  # default Lightning alpha

# ─── FP8 File Locators ───────────────────────────────────────────────────────
FP8_FILES = {
    "diffusion": "diffusion_models/qwen_image_fp8mixed.safetensors",
    "text_encoder": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
}

def find_fp8_file(key: str) -> str | None:
    """Locate one of the three FP8 safetensors on the RunPod volume."""
    rel = FP8_FILES[key]
    for root in COMFY_SEARCH_ROOTS:
        if not root or not os.path.exists(root):
            continue
        candidate = os.path.join(root, rel)
        if os.path.isfile(candidate):
            print(f"[FP8] ✅ {key} → {candidate}")
            return candidate
    return None

def symlink_fp8_for_diffsynth(key: str, src: str):
    """
    DiffSynth's ModelConfig expects files under DIFFSYNTH_MODEL_BASE_PATH.
    Create a stable symlink so we can pass a predictable path to ModelConfig.
    """
    dest_dir = os.path.join(DIFFSYNTH_MODELS_DIR, "comfy_fp8", os.path.dirname(FP8_FILES[key]))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(FP8_FILES[key]))
    if not os.path.exists(dest):
        try:
            os.symlink(src, dest)
        except Exception as e:
            print(f"[FP8] Symlink failed for {key}: {e}")
    return dest

# ─── HF-style Cache Locator (kept for processor fallback) ────────────────────
def find_snapshot_path(model_id, name="Component"):
    search_roots = [HF_CACHE, "/workspace/huggingface-cache", "/runpod-volume/huggingface-cache"]
    hf_name = "models--" + model_id.replace("/", "--")
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        hub_dir = os.path.join(root, "hub") if os.path.isdir(os.path.join(root, "hub")) else root
        model_dir = os.path.join(hub_dir, hf_name)
        if os.path.isdir(model_dir):
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.isdir(snapshots_dir):
                hashes = os.listdir(snapshots_dir)
                if hashes:
                    path = os.path.join(snapshots_dir, hashes[0])
                    print(f"[Cache] ✅ {name} found: {path}")
                    return path
    return None

# ─── Pre-flight Check ─────────────────────────────────────────────────────────
def prepare_cache():
    print(f"\n{'='*60}\n{'[Cache] FP8 READY-CHECK (RTX 5090)':^60}\n{'='*60}")
    missing = []

    # --- FP8 large weights (loaded from RunPod volume, NOT baked) ---
    fp8_paths = {}
    for key in FP8_FILES:
        path = find_fp8_file(key)
        if path:
            fp8_paths[key] = symlink_fp8_for_diffsynth(key, path)
        else:
            print(f"[FP8] ❌ {key} MISSING — expected at {COMFY_SEARCH_ROOTS[0]}/{FP8_FILES[key]}")
            missing.append(f"fp8/{key}")

    # --- Small baked files ---
    lora_path = "/app/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    if os.path.exists(lora_path):
        print(f"[LoRA] ✅ Baked-in LoRA found")
    else:
        print(f"[LoRA] ❌ Baked-in LoRA MISSING")
        missing.append("LoRA")

    proc_path = "/app/models/processor"
    if os.path.isdir(proc_path) and "config.json" in os.listdir(proc_path):
        print(f"[Processor] ✅ Baked-in Processor found")
    else:
        print(f"[Processor] ❌ Baked-in Processor MISSING")
        missing.append("Processor")

    if missing:
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(f"[Cache] ⚠️  Offline mode DISABLED — missing: {missing}")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[Cache] All components ready. Forcing OFFLINE mode.")

    print(f"{'='*60}\n")
    return fp8_paths

# ─── Patching ────────────────────────────────────────────────────────────────
try:
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25
    if not hasattr(_qwen25, 'Qwen2RMSNorm'):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        _qwen25.Qwen2RMSNorm = Qwen2RMSNorm
        print("[Patch] Injected Qwen2RMSNorm into qwen2_5_vl")
except Exception as e:
    print(f"[Patch] Warning: {e}")

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler

# ─── Model Loading ───────────────────────────────────────────────────────────
def load_model():
    global pipe, _loaded_alpha
    start_load = time.time()
    fp8_paths = prepare_cache()

    # Abort early if critical FP8 weights are missing
    for key in ("diffusion", "text_encoder", "vae"):
        if key not in fp8_paths:
            raise RuntimeError(
                f"[Fatal] FP8 weight '{key}' not found. "
                f"Mount the RunPod volume containing Comfy-Org/Qwen-Image_ComfyUI split_files "
                f"and ensure COMFY_CACHE_DIR points to the right root."
            )

    # RTX 5090 has 32 GB VRAM. Reserve ~6 GB for activations.
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    safe_limit = max(16.0, vram_total - 6.0)
    print(f"[VRAM] Total={vram_total:.1f}GB  Safe limit={safe_limit:.1f}GB")

    # FP8 weights load directly from their absolute paths.
    # DiffSynth ModelConfig accepts a direct file path via origin_file_pattern.
    # We use bfloat16 for computation on Blackwell (SM_120 has native BF16 + FP8 tensor cores).
    compute_cfg = {
        "onload_dtype": torch.float8_e4m3fn,  # keep FP8 in VRAM
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    # VAE: always BF16, it's tiny and precision matters for decoding
    vae_cfg = {
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    proc_path = "/app/models/processor"
    # Prefer snapshot processor if available (has sub-configs)
    proc_snap = find_snapshot_path("Qwen/Qwen-Image-Edit-2511", name="Tokenizer Meta")
    if proc_snap and os.path.exists(os.path.join(proc_snap, "processor", "preprocessor_config.json")):
        proc_path = os.path.join(proc_snap, "processor")

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            # Diffusion transformer — FP8 mixed
            ModelConfig(
                model_id="Comfy-Org/Qwen-Image_ComfyUI",
                origin_file_pattern=fp8_paths["diffusion"],
                **compute_cfg,
            ),
            # Text encoder — FP8 scaled
            ModelConfig(
                model_id="Comfy-Org/Qwen-Image_ComfyUI",
                origin_file_pattern=fp8_paths["text_encoder"],
                **compute_cfg,
            ),
            # VAE — kept in BF16 for quality
            ModelConfig(
                model_id="Comfy-Org/Qwen-Image_ComfyUI",
                origin_file_pattern=fp8_paths["vae"],
                **vae_cfg,
            ),
        ],
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit-2511",
            origin_file_pattern=proc_path,
        ),
        vram_limit=safe_limit,
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

# ─── Handler ─────────────────────────────────────────────────────────────────
def parse_alpha(alpha_val):
    if isinstance(alpha_val, (int, float)):
        return float(alpha_val)
    if not isinstance(alpha_val, str):
        return 0.125
    try:
        return float(eval(alpha_val, {"__builtins__": None}, {})) if "/" in alpha_val else float(alpha_val)
    except:
        return 0.125

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
        print(f"Generating I2I | {img_w}x{img_h} | images={len(input_images)} | steps={steps} cfg={cfg_scale} alpha={_loaded_alpha:.4f}")
    else:
        img_w = job_input.get("width", 1024)
        img_h = job_input.get("height", 1024)
        print(f"Generating T2I | {img_w}x{img_h} | steps={steps} cfg={cfg_scale} alpha={_loaded_alpha:.4f}")

    start = time.time()
    try:
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
        print(f"Done: {time.time() - start:.2f}s")
        return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})

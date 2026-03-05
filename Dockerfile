# ───────────────────────────────────────────────────────────────────────────────
# Qwen-Image-Edit-2511 on RunPod Serverless
# Uses DiffSynth-Studio — pure Python, NO CUDA extension compilation needed.
# Optimized for RTX 5090 (32GB VRAM) with ComfyOrg FP8-mixed weights.
# Weights are NOT baked — loaded at runtime from RunPod Network Volume cache.
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# System env
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git wget \
    libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip

# 2. PyTorch for CUDA 12.8 + Blackwell (RTX 5090)
#    torch 2.7+ has native FP8 (torch.float8_e4m3fn) support for GB202
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Python Packages
RUN pip install --no-cache-dir \
    transformers accelerate peft \
    sentencepiece protobuf ftfy \
    Pillow einops \
    runpod huggingface_hub

# 4. DiffSynth-Studio
RUN pip install --no-cache-dir git+https://github.com/modelscope/DiffSynth-Studio.git

# 5. Bake processor files only (tiny, no model weights needed at build time)
RUN mkdir -p /app/models/lora /app/models/processor
RUN echo '{"model_type": "qwen2_5_vl"}' > /app/models/processor/config.json \
    && wget -q -O /app/models/processor/added_tokens.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/added_tokens.json \
    && wget -q -O /app/models/processor/special_tokens_map.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/special_tokens_map.json \
    && wget -q -O /app/models/processor/merges.txt \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/merges.txt \
    && wget -q -O /app/models/processor/vocab.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/vocab.json \
    && wget -q -O /app/models/processor/preprocessor_config.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/preprocessor_config.json \
    && wget -q -O /app/models/processor/tokenizer_config.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/tokenizer_config.json \
    && wget -q -O /app/models/processor/tokenizer.json \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/tokenizer.json \
    && wget -q -O /app/models/processor/chat_template.jinja \
        https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/main/processor/chat_template.jinja

# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

# ─── Runtime Env ──────────────────────────────────────────────────────────────
# RunPod Network Volume cache (set this in your RunPod template)
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV MODELSCOPE_CACHE=/runpod-volume/huggingface-cache

# ComfyOrg FP8 model paths on RunPod volume
# diffusion model  → qwen_image_edit_2511_fp8mixed.safetensors
# text encoder     → qwen_2.5_vl_7b_fp8_scaled.safetensors
ENV COMFY_DIFFUSION_MODEL=/runpod-volume/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
ENV COMFY_TEXT_ENCODER=/runpod-volume/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

ENV DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
ENV DIFFSYNTH_MODEL_BASE_PATH=/app/models

# RTX 5090: large VRAM, less fragmentation pressure — but keep expandable for safety
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Flash Attention 3 (available on Blackwell via torch 2.7)
ENV TORCH_SDPA_BACKEND=flash_attention

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "/app/handler.py"]

# ───────────────────────────────────────────────────────────────────────────────
# Qwen-Image-Edit-2511 on RunPod Serverless
# Uses DiffSynth-Studio — pure Python, NO CUDA extension compilation needed.
# Optimized for RTX 5090 (32GB VRAM, Blackwell SM_120) with FP8 weights.
# Large FP8 weights (diffusion, text_encoder, vae) are loaded at runtime from
# the RunPod network volume cache — NOT baked into the image.
# Small files (LoRA, processor tokenizer) are baked in at build time.
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# System env
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 1. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3-dev git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && pip install --no-cache-dir --upgrade pip

# 2. PyTorch — nightly cu128 required for Blackwell / SM_120 (RTX 5090)
#    Once stable 2.7+ is released, pin to a stable wheel instead.
RUN pip install --no-cache-dir \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Python Packages
RUN pip install --no-cache-dir \
    transformers>=4.51.0 accelerate peft \
    sentencepiece protobuf ftfy \
    Pillow einops \
    runpod huggingface_hub modelscope \
    safetensors

# 4. DiffSynth-Studio
RUN pip install --no-cache-dir git+https://github.com/modelscope/DiffSynth-Studio.git

# ─── Baked Small Files ────────────────────────────────────────────────────────
# Only small, non-GPU-intensive files are baked: LoRA + processor tokenizer.
# The three large FP8 safetensors are loaded at cold-start from the RunPod
# network volume (see COMFY_CACHE_DIR env var and handler.py).
# ─────────────────────────────────────────────────────────────────────────────

RUN mkdir -p /app/models/lora /app/models/processor

# Bake: Lightning 4-step LoRA (small, ~200MB)
RUN wget -q --show-progress -O /app/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

# Bake: Processor / Tokenizer small config files
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

# ─── Environment Variables ────────────────────────────────────────────────────
# RunPod network volume HF cache root
ENV HF_HOME=/workspace/huggingface-cache
ENV MODELSCOPE_CACHE=/workspace/huggingface-cache
# DiffSynth paths
ENV DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
ENV DIFFSYNTH_MODEL_BASE_PATH=/app/models
# Comfy-Org split_files cache root on the RunPod volume.
# Override at container launch if your volume mount point differs.
ENV COMFY_CACHE_DIR=/workspace/ComfyUI/models
# Stay fully offline — weights must be on the volume
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# Blackwell / large-tensor allocator settings
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "/app/handler.py"]

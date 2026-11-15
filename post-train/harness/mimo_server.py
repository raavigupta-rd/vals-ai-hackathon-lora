"""Serve XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (open weights, MIT) with vLLM on Modal GPUs, as a teacher.

Exposes vLLM's OpenAI-compatible API (we use /v1/completions with pre-rendered token ids, rendered with
MiMo's own chat template by harness/mimo_traces.py, so no server-side tool/reasoning parser is involved).

    uv run modal deploy -m harness.mimo_server      # prints the web URL
    MIMO_URL=https://...modal.run uv run python -m harness.mimo_traces ...
"""

import os
import subprocess

import modal

MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
SERVED_NAME = "mimo"
PORT = 8000
GPU = os.environ.get("MIMO_GPU", "H100")
MAX_CONTAINERS = int(os.environ.get("MIMO_MAX_CONTAINERS", 4))

# Official vLLM image: ships the CUDA toolkit, so FlashInfer kernels that are JIT-compiled at startup work.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.33.0",
                              setup_dockerfile_commands=["RUN ln -sf $(command -v python3) /usr/local/bin/python"])
    .entrypoint([])
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("mimo-teacher")


@app.function(image=image, gpu=GPU, timeout=6 * 3600, scaledown_window=10 * 60, max_containers=MAX_CONTAINERS,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache})
@modal.concurrent(max_inputs=96)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "65536", "--gpu-memory-utilization", "0.92", "--enable-prefix-caching",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}', "--max-num-seqs", "96",
        # FlashInfer's GDN (linear-attention) prefill kernel is JIT-compiled and needs nvcc, absent in this image
        "--gdn-prefill-backend", "triton",
    ]
    subprocess.Popen(" ".join(f"'{c}'" if " " in c or "{" in c else c for c in cmd), shell=True)

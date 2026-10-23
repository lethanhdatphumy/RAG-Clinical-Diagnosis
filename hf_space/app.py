"""
Clinical LLM (phimed) — HuggingFace ZeroGPU Space

Loads the fine-tuned phimed GGUF model via llama-cpp-python.

Gradio API endpoint (auto-generated):
  POST /gradio_api/call/generate
  Body: {"data": [prompt, max_tokens, temperature]}
"""

import os
import spaces
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID  = os.getenv("MODEL_REPO",  "lethanhdatphumy/phimed-clinical-gguf")
FILENAME = os.getenv("MODEL_FILE",  "phimed-Q4_K_M.gguf")
N_CTX    = int(os.getenv("N_CTX",  "4096"))
N_GPU    = int(os.getenv("N_GPU_LAYERS", "35"))   # layers to offload to GPU

_llm: Llama | None = None


def _load():
    global _llm
    if _llm is None:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        _llm = Llama(model_path=path, n_ctx=N_CTX, n_gpu_layers=N_GPU, verbose=False)


@spaces.GPU(duration=60)
def generate(prompt: str, max_tokens: int, temperature: float) -> str:
    _load()
    output = _llm(
        prompt,
        max_tokens=max_tokens,
        temperature=max(temperature, 1e-4),
        stop=["</s>", "Question:", "Human:"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=4),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=6),
    title="Clinical LLM — phimed",
    description=f"Fine-tuned clinical model | `{REPO_ID}`",
    api_name="generate",
)

demo.launch(server_name="0.0.0.0")

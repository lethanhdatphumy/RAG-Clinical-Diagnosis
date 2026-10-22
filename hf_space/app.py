"""
Clinical LLM — HuggingFace ZeroGPU Space

Exposes Gradio's built-in REST API:
  POST /run/predict  — {"data": [prompt, max_tokens, temperature], "fn_index": 0}
                       → {"data": ["generated text"]}

Set MODEL_ID in Space Secrets to your fine-tuned model repo.
Default: google/gemma-2-2b-it
"""

import os
import torch
import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2-2b-it")

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _model.eval()


@spaces.GPU(duration=60)
def generate(prompt: str, max_tokens: int, temperature: float) -> str:
    _load()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-4),
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=4),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=6),
    title="Clinical LLM",
    description=f"Fine-tuned clinical language model | Model: `{MODEL_ID}`",
    api_name="generate",
)

demo.launch(server_name="0.0.0.0")

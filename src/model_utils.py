"""Load either a text-only causal LM or a multimodal (image-text-to-text) model,
in both cases exposing input/output token embeddings via the standard HF API.

`get_input_embeddings()` / `resize_token_embeddings()` are defined on PreTrainedModel
and delegate to the text tower even for multimodal wrappers, so the vocab-expansion
and training code can stay model-agnostic. A few exotic wrappers occasionally need a
per-model tweak here; that's the one place to patch if a checkpoint misbehaves.
"""
from __future__ import annotations
import torch


def load_model(repo: str, loader: str = "causal_lm",
               dtype=torch.bfloat16, attn: str | None = None,
               device_map=None, low_cpu_mem_usage=True,
               quantization_config=None):
    kwargs = dict(torch_dtype=dtype, trust_remote_code=True,
                  low_cpu_mem_usage=low_cpu_mem_usage)
    if attn:
        kwargs["attn_implementation"] = attn
    if device_map is not None:
        kwargs["device_map"] = device_map
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    if loader == "image_text_to_text":
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(repo, **kwargs)


def output_embeddings_or_none(model):
    """Return the lm_head weight tensor if the model has untied output embeddings."""
    if getattr(model.config, "tie_word_embeddings", True):
        return None
    oe = model.get_output_embeddings()
    return None if oe is None else oe.weight

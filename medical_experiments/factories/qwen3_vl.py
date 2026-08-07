"""Qwen3-VL adapters for frozen feature export and semantic scoring."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..medical_io import load_array, prepare_model_image


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _imports():
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise RuntimeError(
            "Qwen3-VL requires torch, Pillow, transformers>=4.57, and accelerate"
        ) from exc
    return np, torch, Image, AutoProcessor, Qwen3VLForConditionalGeneration


def _dtype(torch: Any, value: str):
    name = str(value).lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in choices:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    return choices[name]


def _pil_image(image: Any, image_size: int):
    np, _torch, Image, _processor, _model = _imports()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3):
        raise ValueError("Qwen3-VL input must be HxW, HxWx1, or HxWx3")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = np.clip(array, 0.0, 1.0)
    result = Image.fromarray(np.round(array * 255.0).astype("uint8"), mode="RGB")
    if int(image_size) > 0:
        result = result.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    return result


def _prompt_inputs(processor: Any, prompt: str, image: Any = None):
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs: Dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if image is not None:
        kwargs["images"] = [image]
    return processor(**kwargs)


def _load_components(kwargs: Mapping[str, Any]):
    _np, torch, _Image, AutoProcessor, ModelClass = _imports()
    model_id = str(kwargs.get("model_id", DEFAULT_MODEL_ID))
    dtype = _dtype(torch, str(kwargs.get("dtype", "bfloat16")))
    load_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    attention = kwargs.get("attn_implementation", "sdpa")
    if attention:
        load_kwargs["attn_implementation"] = str(attention)
    revision = kwargs.get("revision")
    if revision:
        load_kwargs["revision"] = str(revision)
    model = ModelClass.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return model, processor, torch


def build_feature_bundle(kwargs: Mapping[str, Any]):
    """Build the feature-export protocol used by experiment one.

    The real config fixes images to 512x512. Qwen3-VL therefore exposes a
    32x32 pre-merger vision grid and a 16x16 post-merger/LLM image-token grid.
    """

    model, processor, _torch = _load_components(kwargs)
    image_size = int(kwargs.get("image_size", 512))
    prompt = str(
        kwargs.get(
            "prompt",
            "Inspect this medical image for localized visual abnormalities. "
            "Do not use patient labels or a reference report.",
        )
    )
    llm_layers = [int(value) for value in kwargs.get("llm_layers", [18, 36])]
    state: Dict[str, Any] = {}

    def preprocess(image: Any, record: Mapping[str, Any]):
        del record
        prepared = _prompt_inputs(processor, prompt, _pil_image(image, image_size))
        state["input_ids"] = prepared["input_ids"]
        state["image_grid_thw"] = prepared.get("image_grid_thw")
        return prepared

    def forward(model_instance: Any, prepared: Mapping[str, Any]):
        return model_instance(
            **prepared, output_hidden_states=True, use_cache=False, return_dict=True
        )

    def extract_output(output: Any):
        hidden_states = output.hidden_states
        image_token_id = int(model.config.image_token_id)
        mask = state["input_ids"].reshape(-1) == image_token_id
        expected = None
        grid = state.get("image_grid_thw")
        merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
        if grid is not None:
            expected = int(grid.detach().cpu().prod(dim=-1).sum().item()) // (merge * merge)
        if expected is not None and int(mask.sum().item()) != expected:
            raise RuntimeError(
                "Qwen image-token count mismatch: got %d, expected %d"
                % (int(mask.sum().item()), expected)
            )
        result = {}
        for layer in llm_layers:
            index = layer if layer >= 0 else len(hidden_states) + layer
            if index < 0 or index >= len(hidden_states):
                raise ValueError(
                    "LLM hidden-state index %d is outside [0, %d)"
                    % (layer, len(hidden_states))
                )
            flat = hidden_states[index].reshape(-1, hidden_states[index].shape[-1])
            result["qwen_llm_%02d" % layer] = flat[mask.to(flat.device)]
        return result

    return {
        "model": model,
        "preprocess": preprocess,
        "forward": forward,
        "extract_output": extract_output,
    }


def _knowledge_descriptions(kwargs: Mapping[str, Any]) -> Dict[str, str]:
    raw = kwargs.get("knowledge", {})
    knowledge_file = kwargs.get("knowledge_file")
    if knowledge_file:
        with Path(str(knowledge_file)).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("knowledge or knowledge_file must contain a JSON mapping")
    return {str(key): str(value) for key, value in raw.items()}


def _deranged_descriptions(
    candidates: Sequence[str], descriptions: Mapping[str, str], seed: int
) -> Dict[str, str]:
    names = [str(value) for value in candidates]
    values = [str(descriptions.get(name, name)) for name in names]
    if len(names) < 2:
        return dict(zip(names, values))
    rng = random.Random(int(seed))
    order = list(range(len(names)))
    for _ in range(100):
        rng.shuffle(order)
        if all(index != order[index] for index in range(len(names))):
            return {name: values[order[index]] for index, name in enumerate(names)}
    return {name: values[(index + 1) % len(names)] for index, name in enumerate(names)}


def _append_candidate(base: Mapping[str, Any], candidate_ids: Any, torch: Any):
    prepared = dict(base)
    input_ids = base["input_ids"]
    candidate_ids = candidate_ids.to(input_ids.device).reshape(1, -1)
    prepared["input_ids"] = torch.cat([input_ids, candidate_ids], dim=1)
    extension = torch.ones_like(candidate_ids)
    prepared["attention_mask"] = torch.cat(
        [base["attention_mask"], extension.to(base["attention_mask"].dtype)], dim=1
    )
    if "mm_token_type_ids" in base:
        text_types = torch.zeros_like(candidate_ids, dtype=base["mm_token_type_ids"].dtype)
        prepared["mm_token_type_ids"] = torch.cat(
            [base["mm_token_type_ids"], text_types], dim=1
        )
    return prepared, int(input_ids.shape[1]), int(candidate_ids.shape[1])


def make_semantic_scorer(kwargs: Mapping[str, Any]):
    """Return a four-condition Qwen3-VL candidate log-likelihood scorer."""

    model, processor, torch = _load_components(kwargs)
    device = str(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image_size = int(kwargs.get("image_size", 512))
    preprocessing = dict(kwargs.get("preprocessing", {}))
    descriptions = _knowledge_descriptions(kwargs)
    seed = int(kwargs.get("seed", 3407))

    def score(
        record: Mapping[str, Any],
        candidates: Sequence[str],
        evidence_present: bool,
        knowledge_intact: bool,
    ) -> Dict[str, float]:
        candidate_names = [str(value) for value in candidates]
        knowledge = (
            {name: descriptions.get(name, name) for name in candidate_names}
            if knowledge_intact
            else _deranged_descriptions(candidate_names, descriptions, seed)
        )
        lines = ["Candidate medical knowledge:"] + [
            "- %s: %s" % (name, knowledge[name]) for name in candidate_names
        ]
        lines.extend(
            [
                "Organ: %s" % record.get("organ", "unknown"),
                "Region: %s" % record.get("region", "not provided"),
                "Choose the single most likely finding from: %s." % ", ".join(candidate_names),
                "Answer with the exact finding name only.",
            ]
        )
        image = None
        if evidence_present:
            if not record.get("image"):
                raise ValueError("evidence-present semantic scoring requires record.image")
            raw = load_array(Path(str(record["image"])), key=record.get("image_key"))
            local_preprocessing = dict(preprocessing)
            if record.get("slice_index") is not None:
                local_preprocessing["slice_index"] = int(record["slice_index"])
            prepared_image = prepare_model_image(
                raw, str(record.get("modality", "unknown")), local_preprocessing
            )
            image = _pil_image(prepared_image, image_size)
        base = _prompt_inputs(processor, "\n".join(lines), image=image).to(device)
        result: Dict[str, float] = {}
        with torch.inference_mode():
            for candidate in candidate_names:
                ids = processor.tokenizer(
                    candidate, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                full, prompt_length, candidate_length = _append_candidate(base, ids, torch)
                output = model(**full, use_cache=False, return_dict=True)
                log_probs = torch.log_softmax(output.logits.float(), dim=-1)
                token_ids = full["input_ids"][0, prompt_length : prompt_length + candidate_length]
                positions = torch.arange(
                    prompt_length - 1,
                    prompt_length + candidate_length - 1,
                    device=log_probs.device,
                )
                result[candidate] = float(
                    log_probs[0, positions, token_ids].mean().detach().cpu().item()
                )
        return result

    return {"score": score}

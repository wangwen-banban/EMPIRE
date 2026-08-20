import copy
import os
import transformers
import torch


def _resolve_device_map(device_map):
    if device_map is None:
        return "cpu"
    if isinstance(device_map, str):
        value = device_map.strip()
        if value == "":
            return "cpu"
        if value.lower() in {"none", "null"}:
            return None
        if value.lower() == "cuda":
            return {"": "cuda"}
        return value
    return device_map


def build_vlm(vlm_config):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=_resolve_device_map(vlm_config.get("device_map") or os.environ.get("VLM_DEVICE_MAP")),
            # attn_implementation="eager",
            # revision="bfloat16",
        )
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    elif model_name == "qwen2_5vl":
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
        )

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return processor, model

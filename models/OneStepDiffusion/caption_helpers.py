"""Image-to-text helpers for VLM prompt extractors."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW image tensor in [0, 1] to a PIL Image."""
    array = tensor.detach().clamp(0, 1).float().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def _resolve_vlm_dtype(model_name: str, device, dtype):
    """Pick a safe inference dtype for the requested VLM."""
    model_name_lower = model_name.lower()
    if "gemma" in model_name_lower:
        # Gemma 3 produces empty captions in float16; bfloat16 is the supported path.
        if device != "cpu" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return dtype


def _move_vlm_inputs(inputs: dict, device, dtype) -> dict:
    moved = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "pixel_values" in moved:
        moved["pixel_values"] = moved["pixel_values"].to(dtype=dtype)
    return moved


def _load_chat_vlm(model_name: str, device, dtype):
    model_name_lower = model_name.lower()
    dtype = _resolve_vlm_dtype(model_name, device, dtype)
    if "gemma" in model_name_lower:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
        ).to(device)
    else:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_cls = AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)

    model.eval()
    return model, processor, dtype


class ImageCaptioner:
    def __init__(
        self,
        model_name: str,
        instruction: str,
        device,
        dtype=torch.float16,
        max_new_tokens: int = 96,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._model_dtype = dtype

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._model, self._processor, self._model_dtype = _load_chat_vlm(
            self.model_name,
            self.device,
            self.dtype,
        )

    def _caption_one(self, pil_image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_vlm_inputs(inputs, self.device, self._model_dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[0][input_len:]
        return self._processor.decode(generated_ids, skip_special_tokens=True).strip()

    def caption_batch(self, images: torch.Tensor) -> list[str]:
        self._ensure_loaded()
        bsz = images.shape[0]
        return [self._caption_one(tensor_to_pil(images[i])) for i in range(bsz)]

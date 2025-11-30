from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Optional, cast

from .base import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MT_DIR = PROJECT_ROOT / "models" / "lfm2-enjp"

class LFM2EnJpTranslator(BaseModel):
    """
    LFM2-350M-ENJP-MTのラッパ
    日英、英日を行う
    """

    def __init__(self) -> None:
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None

    def load(self):
        if self.model is not None:
            return

        model_id = str(MT_DIR) if MT_DIR.exists() else "LiquidAI/LFM2-350M-ENJP-MT"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = "auto",
            device_map = "auto",
        )

    def _generate(self, system_prompt: str, text: str) -> str:
        self.load()
        
        assert self.tokenizer is not None
        assert self.model is not None

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.5,
            min_p=0.1,
            top_p=1.0,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )
        
        new_tokens = output[0, input_ids.shape[-1]:]
        out = self.tokenizer.decode(new_tokens, skip_special_toknes=True)
        return out.strip()

    def infer(self, text: str, direction: str = "en2ja") -> str:
        """
        direction:
            - "en2ja": 英日
        """
        
        if direction == "en2ja":
            system_prompt = "Translate to Japanese."
        elif direction == "ja2en":
            system_prompt = "Translate to English."
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return self._generate(system_prompt, text)
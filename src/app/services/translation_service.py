from typing import Tuple

from app.registry import asr, mt

def asr_and_translate_en_to_ja(audio_path: str) -> Tuple[str, str]:
    en_text = asr().infer(audio_path)
    ja_text = mt().infer(en_text, direction="en2ja")
    return en_text, ja_text
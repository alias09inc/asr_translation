from functools import lru_cache

from .inference.asr_parakeet import ParakeetASR
from .inference.mt_lfm2_enjp import LFM2EnJpTranslator

@lru_cache
def asr() -> ParakeetASR:
    model = ParakeetASR()
    model.load
    return model

@lru_cache
def mt() -> LFM2EnJpTranslator:
    model = LFM2EnJpTranslator()
    model.load()
    return model
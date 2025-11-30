from pathlib import Path
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

from .base import BaseModel

PROJECT_ROOT = Path(__file__).reslove().parents[3]
ASR_DIR = PROJECT_ROOT / "models" / "parakeet-0.6b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ParakeetASR(BaseModel):
    """
    英語 ASRのラッパ
    """

    def __init__(self) -> None:
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        model_id = str(ASR_DIR) if ASR_DIR.exists() else "nvidia/parakeet-ctc-0.6b"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def _prepare_inputs(self, audio_path: str):
        self.load()

        audio, sr = sf.read(audio_path)
        target_sr = self.processor.feature_extractor.sampling_rate

        if sr != target_sr:
            raise ValueError(f"Parakeet expects {target_sr} Hz, got {sr}")

        if audio.ndim == 2:
            # stereo -> mono
            audio = audio.mean(axis=1)

        inputs = self.processor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def infer(self, audio_path: str) -> str:
        """
        audio_path: 16kHz mono wav
        return: 認識した英語テキスト
        """
        inputs = self._prepare_inputs(audio_path)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip()

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from .base import BaseModel

try:
	import nemo.collections.asr as nemo_asr  # type: ignore[import]
except Exception:  # pragma: no cover
	# If nemo is not installed, set a sentinel to avoid import errors.
	nemo_asr = None  # type: ignore[assignment]

class ParakeetASR(BaseModel):
    """
    Multilingual ASR: nvidia/parakeet-tdt-0.6b-v3 のラッパ

    - 16kHz モノラルの .wav / .flac を想定（NeMo 側の仕様に準拠）
    - 言語自動判定 + 句読点付きテキストを返す
    - timestamps=True の場合は NeMo の timestamp 情報も返す
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            Hugging Face / NGC のモデル名。
            デフォルトは parakeet-tdt-0.6b-v3。
        device:
            例: "cuda", "cpu"。
            None の場合は NeMo のデフォルト挙動に任せる。
        """
        self.model_name = model_name
        self.device = device
        self._model: nemo_asr.models.ASRModel | None = None

    # BaseModel のインターフェイス
    def load(self) -> None:
        """
        NeMo ASRModel を遅延ロードする。
        - すでにロード済みなら何もしない。
        """
        if self._model is not None:
            return

        # HF から自動ダウンロード（ローカルキャッシュも利用される）
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )

        if self.device is not None:
            # NeMo モデルも .to(device) が使える（内部は PyTorch Lightning モジュール）
            self._model = self._model.to(self.device)

        # ここで self._model.transcribe(...) が使える状態になる

    def _call_transcribe(
        self,
        audio_path: str,
        timestamps: bool,
    ) -> Any:
        """
        NeMo の transcribe を呼ぶラッパ。
        返り値は NeMo の ASRResult / str などバージョン依存なので、
        infer() 側で型を吸収する。
        """
        self.load()
        assert self._model is not None

        # parakeet-tdt-0.6b 系の公式コードと同じ呼び方 
        outputs = self._model.transcribe(
            [audio_path],
            timestamps=timestamps,
        )
        return outputs

    def infer(
        self,
        audio_path: str,
        *,
        timestamps: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any] | None]]:
        """
        audio_path の音声を文字起こしする。

        Parameters
        ----------
        audio_path:
            16kHz mono の .wav / .flac ファイルパスを想定。
        timestamps:
            True の場合、(text, timestamp_dict) を返す。
            False の場合、text のみを返す。

        Returns
        -------
        text or (text, timestamp_dict)
        """
        outputs = self._call_transcribe(audio_path, timestamps=timestamps)

        if not outputs:
            # 何も返ってこなかった場合の保険
            return ("", None) if timestamps else ""

        first = outputs[0]

        # パターン1: NeMo 新 API (ASRResult オブジェクト)
        #   - first.text でテキスト
        #   - first.timestamp["word"/"segment"/"char"] でタイムスタンプ
        if hasattr(first, "text"):
            text = first.text
            ts = getattr(first, "timestamp", None)
            if timestamps:
                # ts は None か、"word"/"segment"/"char" を持つ dict と想定
                return text, ts
            else:
                return text

        # パターン2: 古い NeMo では transcribe() が単純な str を返すケース
        if isinstance(first, str):
            return (first, None) if timestamps else first

        # パターン3: 予期しない型（念のため文字列化）
        text = str(first)
        return (text, None) if timestamps else text
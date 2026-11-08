"""Polish speech synthesis with XTTS-v2, cloning the voice from a short sample."""

from __future__ import annotations

import re

import numpy as np

from .asr import frame_energy_db
from .segments import split_sentences

DEFAULT_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"


def trim_silence(
    audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0, pad_s: float = 0.03
) -> np.ndarray:
    """Cut leading/trailing silence so the measured duration is the speech itself."""
    if audio.size == 0:
        return audio
    frame_s = 0.01
    db = frame_energy_db(audio, sample_rate, frame_s)
    idx = np.flatnonzero(db > db.max() + threshold_db)
    if idx.size == 0:
        return audio
    hop = int(sample_rate * frame_s)
    pad = int(sample_rate * pad_s)
    a = max(0, idx[0] * hop - pad)
    b = min(len(audio), (idx[-1] + 1) * hop + pad)
    return audio[a:b]


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into pieces XTTS can say in one go (at most ``limit`` characters).

    Sentences first, then clause boundaries (commas, semicolons), then words.
    XTTS has its own splitter, but it needs spaCy; this one has no dependencies.
    """
    pieces: list[str] = []
    for sent in split_sentences(text):
        if len(sent) <= limit:
            pieces.append(sent)
            continue
        current = ""
        for part in re.split(r"(?<=[,;:–])\s+", sent):
            if len(part) > limit:  # no usable clause boundary: split on words
                words, part = part.split(), ""
                for w in words:
                    if part and len(part) + 1 + len(w) > limit:
                        pieces.append(part)
                        part = w
                    else:
                        part = f"{part} {w}".strip()
            if current and len(current) + 1 + len(part) > limit:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            pieces.append(current)
    return pieces


class XTTSSynthesizer:
    """Thin wrapper around Coqui XTTS-v2 that computes the speaker latents once.

    On first use Coqui asks you to accept the XTTS-v2 model licence (CPML,
    non-commercial). Set ``COQUI_TOS_AGREED=1`` to accept it non-interactively.
    """

    def __init__(
        self,
        voice_sample: str,
        device: str = "cpu",
        model_name: str = DEFAULT_XTTS,
        temperature: float = 0.7,
        pause_s: float = 0.15,
    ):
        from TTS.api import TTS

        self.api = TTS(model_name).to(device)
        self.model = self.api.synthesizer.tts_model
        self.sample_rate = int(self.model.config.audio.output_sample_rate)
        self.temperature = temperature
        self.pause_s = pause_s
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(voice_sample)]
        )

    def synthesize(self, text: str, language: str = "pl", seed: int | None = 0) -> np.ndarray:
        """Synthesise ``text``; long text is split into sentences/clauses joined by short pauses."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        limit = self.model.tokenizer.char_limits.get(language, 250)
        pause = np.zeros(int(self.pause_s * self.sample_rate), dtype=np.float32)
        parts = []
        for piece in chunk_text(text, limit) or [text]:
            with torch.inference_mode():
                out = self.model.inference(
                    piece,
                    language,
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                    temperature=self.temperature,
                    enable_text_splitting=False,
                )
            wav = out["wav"]
            if hasattr(wav, "cpu"):
                wav = wav.cpu().numpy()
            if parts:
                parts.append(pause)
            parts.append(trim_silence(np.asarray(wav, dtype=np.float32).reshape(-1), self.sample_rate))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

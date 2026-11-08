"""English → Polish translation with NLLB-200, one sentence at a time."""

from __future__ import annotations

from .segments import Segment, split_sentences

DEFAULT_NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"en": "eng_Latn", "pl": "pol_Latn"}


class NLLBTranslator:
    def __init__(
        self,
        model_id: str = DEFAULT_NLLB,
        device: str = "cpu",
        src_lang: str = "en",
        tgt_lang: str = "pl",
        num_beams: int = 4,
        length_penalty: float = 1.5,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=NLLB_CODES[src_lang])
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.tgt_code = NLLB_CODES[tgt_lang]
        # Plain beam search (length_penalty=1.0) tends to drop the end of long
        # sentences (e.g. a final "..., and that's what we hope to get out of
        # this."); a length penalty > 1 keeps the whole sentence.
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.batch_size = batch_size

    def translate(self, sentences: list[str], max_length: int = 256) -> list[str]:
        """Translate a list of sentences, returning one translation per input.

        Each sentence is its own item in the batch. Note: the old notebook took
        only ``batch_decode(...)[0]``, which kept the first sentence of the batch
        and looked as if NLLB dropped the rest.
        """
        import torch

        results: list[str] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_code),
                    max_length=max_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                )
            results.extend(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
        return [r.strip() for r in results]


def translate_segments(segments: list[Segment], translator: NLLBTranslator) -> list[Segment]:
    """Fill ``text_pl`` of every segment. Segments are split into sentences first."""
    per_segment = [split_sentences(s.text_en) for s in segments]
    flat = [sent for sents in per_segment for sent in sents]
    translated = iter(translator.translate(flat)) if flat else iter(())
    for seg, sents in zip(segments, per_segment):
        seg.text_pl = " ".join(next(translated) for _ in sents)
    return segments

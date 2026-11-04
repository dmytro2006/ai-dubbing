# ai-dubbing

Automatic **English → Polish video dubbing** that keeps the dubbed speech **in sync with the original video** and in the **original speaker's voice**.

> 🚧 **Work in progress.** Each stage of the pipeline has been prototyped on its own (see [`examples/`](examples/)). End-to-end integration and evaluation are the next steps.

University R&D project (*Studio projektowe 1*, AGH University of Krakow, winter semester 2026/27) by **Dmytro Harasiuk** and **Maksymilian Roman**.

## How it works

```
video ─► FFmpeg ─► Whisper ─► NLLB-200 ─► length adaptation ─► XTTS-v2 ─► timing sync ─► FFmpeg ─► dubbed video
         audio     EN text +   EN → PL     LLM paraphrasing     Polish      fit segments    new audio +
                   timestamps  translation (method M2 only)     cloned      to original     original picture
                                                                voice       time slots
```

| Stage | Model | Role |
|---|---|---|
| Speech recognition | [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | English transcript with segment timestamps |
| Translation | [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | English → Polish, sentence by sentence |
| Speech synthesis | [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (Coqui TTS) | Polish speech cloned from a short voice sample |
| Length adaptation | Qwen (planned) | Shorter or longer paraphrases so the speech fits its time slot |

Pre-trained models are used as-is; no model is trained from scratch. The project's contribution is the processing pipeline, the timing method, the dataset and the evaluation.

## Research question

Polish sentences often run longer than the English originals, so dubbed speech drifts out of sync with the picture. We compare three ways to handle it:

- **M0 – no correction:** recognise, translate, synthesise.
- **M1 – time-stretching:** M0, then speed up or slow down the generated speech without changing its pitch.
- **M2 – text adaptation + limited time-stretching:** an LLM rewrites the translation to be shorter or longer. After synthesis the real duration is measured and the text is revised (at most 3 iterations), followed by a bounded tempo correction.

**Hypothesis:** M2 reduces timing error compared with no correction, while avoiding the audible artefacts of heavy time-stretching.

### Evaluation (planned)

- **Dataset:** 15 self-recorded clips of 10–30 s, at least 3 speakers, with transcripts, reference translations and segment boundaries. 5 clips are for development and 10 are held out for testing. Voice samples are kept separate from the test clips, and every speaker gives consent.
- **Automatic metrics:**
  - relative duration error and segment-boundary error;
  - amount of tempo correction applied;
  - processing time;
  - WER of the synthesised speech against the text sent to synthesis ([jiwer](https://github.com/jitsi/jiwer));
  - voice similarity via ECAPA-TDNN speaker embeddings.
- **Listening test:** at least 5 bilingual raters score meaning, naturalness and voice similarity on a 1–5 scale, in random order.

**Scope:** short single-speaker clips with no overlapping speech or dominant music. Lip-sync, music separation and emotion transfer are out of scope.

## Findings so far

- **NLLB-200 drops sentences.** For English → Polish, NLLB-200 returns only the *first* sentence when given several at once. Other language pairs handle multiple sentences, and the extra context improves the result. For now the pipeline translates EN → PL **one sentence at a time**. Next we will check whether surrounding sentences can be passed as context without being translated.
- **GPU memory.** NLLB-200 distilled 600M needs about 5 GB of VRAM. The 3.3B variant needs more than 8 GB, so on an 8 GB laptop GPU it has to run on the CPU, taking tens of seconds per sentence.

## Status

- [x] Project plan, scope and methodology
- [x] Environment setup (uv, PyTorch with CUDA)
- [x] Whisper prototype: transcription with timestamps
- [x] NLLB-200 prototype: translation
- [x] XTTS-v2 prototype: Polish voice cloning
- [ ] Dataset recording and annotation
- [ ] End-to-end pipeline (audio extraction → … → dubbed video)
- [ ] Methods M1 (time-stretching) and M2 (LLM length adaptation)
- [ ] Evaluation and report

## Getting started

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is recommended. On macOS, uv installs the CPU build of PyTorch.

```bash
git clone https://github.com/dmytro2006/ai-dubbing
cd ai-dubbing
uv sync
```

Then open the notebooks in [`examples/`](examples/) with the project's `.venv` as the kernel, for example in VS Code. Each notebook runs one stage:

- `whisper.ipynb`: speech recognition
- `nllb.ipynb`: translation
- `coqui.ipynb`: voice-cloned speech synthesis

## Tech stack

Python · PyTorch · Hugging Face Transformers · Coqui TTS · FFmpeg · uv

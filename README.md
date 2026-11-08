# ai-dubbing

Automatic **English → Polish video dubbing** that keeps the dubbed speech **in sync with the original video** and in the **original speaker's voice**.

> 🚧 **Work in progress.** The end-to-end pipeline with methods **M0** and **M1** works from the command line (`ai-dubbing video.mp4`). Method M2 (LLM length adaptation) and the full evaluation are next.

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

- **M0 – no correction:** recognise, translate, synthesise. Each Polish segment starts at the original start time, or right after the previous one if that is still playing, so delays add up.
- **M1 – time-stretching:** M0, then speed up or slow down each generated segment to the length of its original slot without changing its pitch (FFmpeg `atempo`; optional limits `--max-stretch` / `--min-stretch`).
- **M2 – text adaptation + limited time-stretching:** an LLM rewrites the translation to be shorter or longer. After synthesis the real duration is measured and the text is revised (at most 3 iterations), followed by a bounded tempo correction.

**Hypothesis:** M2 reduces timing error compared with no correction, while avoiding the audible artefacts of heavy time-stretching.

### Evaluation (planned)

- **Dataset:** 15 clips of 10–30 s from 6 speakers (3 F / 3 M), cut from public-domain NASA interviews, with reference transcripts and segment boundaries from NASA's closed captions. 5 clips are for development and 10 are held out for testing. Voice samples come from parts of the videos not used in any clip. See [`data/README.md`](data/README.md).
- **Automatic metrics:**
  - relative duration error and segment-boundary error;
  - amount of tempo correction applied;
  - processing time;
  - WER of the synthesised speech against the text sent to synthesis ([jiwer](https://github.com/jitsi/jiwer));
  - voice similarity via ECAPA-TDNN speaker embeddings.
- **Listening test:** at least 5 bilingual raters score meaning, naturalness and voice similarity on a 1–5 scale, in random order.

**Scope:** short single-speaker clips with no overlapping speech or dominant music. Lip-sync, music separation and emotion transfer are out of scope.

## Findings so far

- **NLLB-200 does not drop sentences; the notebook did.** The earlier note said NLLB returns only the first sentence for English → Polish. The cause was `tokenizer.batch_decode(...)[0]` in `examples/nllb.ipynb`: a list of sentences is a batch, and `[0]` keeps only the first translation (the Ukrainian example in the notebook shows the same thing). The pipeline translates sentence by sentence and keeps every result.
- **Beam search cuts long sentences short.** With the default `length_penalty=1.0`, NLLB-200 600M sometimes drops the end of a long sentence (*"…, and that's what we really hope to get out of this."* disappeared). This also makes the Polish text artificially shorter, which would distort the timing comparison. The pipeline uses `num_beams=4, length_penalty=1.5`, which keeps the whole sentence.
- **NLLB-200 600M quality limits.** It translates the acronym *GOLD* as *Złoto* and *take off* as *wycofać się*. This is worth noting in the paper's limitations, and M2's LLM pass could fix some of it.
- **GPU memory.** NLLB-200 distilled 600M needs about 5 GB of VRAM. The 3.3B variant needs more than 8 GB, so on an 8 GB laptop GPU it has to run on the CPU, taking tens of seconds per sentence. XTTS-v2 ran out of memory when loaded next to the other notebooks, so the pipeline loads one model at a time and frees it before the next stage.

## Status

- [x] Project plan, scope and methodology
- [x] Environment setup (uv, PyTorch with CUDA)
- [x] Whisper prototype: transcription with timestamps
- [x] NLLB-200 prototype: translation
- [x] XTTS-v2 prototype: Polish voice cloning
- [x] Dataset: 15 clips (5 dev / 10 test) with metadata and caption-based references
- [x] End-to-end pipeline (audio extraction → … → dubbed video) with a CLI
- [x] Method M0 (no correction) and M1 (time-stretching) + timing metrics
- [ ] Manual check of reference transcripts, reference translations
- [ ] Method M2 (LLM length adaptation with Qwen)
- [ ] WER and voice-similarity metrics, listening test, report

## Getting started

Requires Python 3.14, [uv](https://docs.astral.sh/uv/) and [FFmpeg](https://ffmpeg.org/) (`brew install ffmpeg` / `sudo apt install ffmpeg`). A CUDA GPU is recommended; everything also runs on the CPU, just slower (about 5 minutes per 20-second clip on 2 CPU cores). On macOS, uv installs the CPU build of PyTorch.

```bash
git clone https://github.com/dmytro2006/ai-dubbing
cd ai-dubbing
uv sync
```

The first run downloads the three models (about 6 GB) and asks you to accept the XTTS-v2 licence (Coqui Public Model License, non-commercial). Set `COQUI_TOS_AGREED=1` to skip the prompt.

### Dub a video

```bash
uv run ai-dubbing video.mp4                        # M0 and M1, voice cloned from the video itself
uv run ai-dubbing video.mp4 --voice speaker.wav    # clone from a separate sample (6–15 s of clean speech)
uv run ai-dubbing video.mp4 -m m1 --max-stretch 1.3
uv run ai-dubbing a.mp4 b.mp4 -o results/          # several files, models loaded once
uv run ai-dubbing video.mp4 --reuse -m m1 --max-stretch 1.2   # re-render without re-running the models
```

Results go to `output/<video name>/`:

| File | Content |
|---|---|
| `<name>.m0.mp4`, `<name>.m1.mp4` | dubbed video for each method (picture untouched; if the dub runs longer than the video, the last frame is held) |
| `<name>.m0.wav`, `<name>.m1.wav` | the dubbed audio track |
| `<name>.en.srt`, `<name>.pl.srt` | transcript and translation with segment times |
| `segments.json` | segments: original slot, English and Polish text, synthesised duration |
| `report.json` | settings, processing time per stage and timing metrics per method |
| `tts/seg_*.wav` | synthesised Polish speech per segment before synchronisation |

Run `uv run ai-dubbing --help` for all options.

### Dataset and batch runs

```bash
uv run python scripts/build_dataset.py          # download the NASA sources, cut 15 clips + voice samples
uv run python scripts/run_dataset.py --split dev   # dub all dev clips, write output/dev/summary.csv
uv run pytest                                    # fast tests, no models needed
```

### Notebooks

[`examples/`](examples/) holds the first single-stage prototypes (`whisper.ipynb`, `nllb.ipynb`, `coqui.ipynb`). Open them with the project's `.venv` as the kernel.

## Project structure

```
src/ai_dubbing/
  cli.py         command line (ai-dubbing ...)
  pipeline.py    runs the stages over all inputs, one model in memory at a time
  asr.py         Whisper transcription + speech-boundary refinement
  translate.py   NLLB-200 translation, sentence by sentence
  tts.py         XTTS-v2 voice-cloned synthesis
  sync.py        M0 / M1 track assembly (time-stretching via FFmpeg atempo)
  metrics.py     timing metrics (duration / boundary error, stretch factor)
  media.py       FFmpeg helpers
  segments.py    segment data structure, sentence splitting/merging, SRT
scripts/         dataset build and batch evaluation
data/            dataset description (media are rebuilt, not stored in git)
tests/           pytest
```

## Tech stack

Python · PyTorch · Hugging Face Transformers · Coqui TTS · FFmpeg · uv

"""End-to-end dubbing pipeline: video (EN) → dubbed video (PL), methods M0 and M1.

Stages run one after another over all input files (stage-major), and each model
is released before the next one is loaded, so the three models never sit in
memory together (they do not fit on an 8 GB GPU at the same time).
"""

from __future__ import annotations

import gc
import json
import logging
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import media
from .asr import ASR_SAMPLE_RATE, DEFAULT_WHISPER, WhisperASR, refine_boundaries
from .metrics import timing_metrics
from .segments import Segment, load_segments, merge_into_sentences, save_segments, write_srt
from .sync import build_track
from .translate import DEFAULT_NLLB, NLLBTranslator, translate_segments
from .tts import DEFAULT_XTTS, XTTSSynthesizer

log = logging.getLogger(__name__)

VOICE_SAMPLE_RATE = 22050


@dataclass
class DubbingConfig:
    methods: tuple[str, ...] = ("m0", "m1")
    output_dir: Path = Path("output")
    voice_sample: Path | None = None
    device: str = "auto"
    whisper_model: str = DEFAULT_WHISPER
    nllb_model: str = DEFAULT_NLLB
    tts_model: str = DEFAULT_XTTS
    min_stretch: float | None = None
    max_stretch: float | None = None
    seed: int = 0
    reuse: bool = False
    merge_max_gap: float = 0.8
    merge_max_duration: float = 15.0
    voice_sample_seconds: float = 12.0


@dataclass
class Job:
    """State of one input file as it moves through the stages."""

    input: Path
    work_dir: Path
    is_video: bool = True
    duration: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    voice_sample: Path | None = None
    timings: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return self.input.stem

    @property
    def segments_path(self) -> Path:
        return self.work_dir / "segments.json"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _release_memory() -> None:
    gc.collect()
    try:  # give freed CPU memory back to the OS (glibc keeps it otherwise)
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def auto_voice_sample(job: Job, max_seconds: float) -> Path:
    """Cut a voice sample for cloning from the input itself (longest speech segments first)."""
    path = job.work_dir / "voice_sample.wav"
    picked, total = [], 0.0
    for seg in sorted(job.segments, key=lambda s: s.duration, reverse=True):
        if total >= max_seconds:
            break
        picked.append(seg)
        total += seg.duration
    picked.sort(key=lambda s: s.start)
    parts = [
        media.load_audio(job.input, VOICE_SAMPLE_RATE, start=s.start, end=s.end) for s in picked
    ]
    gap = np.zeros(int(0.2 * VOICE_SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([x for p in parts for x in (p, gap)]) if parts else np.zeros(1)
    media.save_wav(path, audio, VOICE_SAMPLE_RATE)
    return path


def _can_reuse(job: Job) -> bool:
    if not job.segments_path.exists():
        return False
    segs = load_segments(job.segments_path)
    return bool(segs) and all(s.text_pl and s.tts_path and Path(s.tts_path).exists() for s in segs)


def run(inputs: list[Path], cfg: DubbingConfig) -> list[Job]:
    media.require_ffmpeg()
    device = resolve_device(cfg.device)
    log.info("Device: %s", device)

    jobs = []
    for path in inputs:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        work = Path(cfg.output_dir) / path.stem
        (work / "tts").mkdir(parents=True, exist_ok=True)
        job = Job(input=path, work_dir=work, is_video=media.has_video(path), duration=media.duration(path))
        jobs.append(job)

    todo = []
    for job in jobs:
        if cfg.reuse and _can_reuse(job):
            job.segments = load_segments(job.segments_path)
            log.info("[%s] reusing transcription, translation and speech from %s", job.stem, job.work_dir)
        else:
            todo.append(job)

    # 1) Speech recognition --------------------------------------------------
    if todo:
        log.info("Loading Whisper (%s)...", cfg.whisper_model)
        asr = WhisperASR(cfg.whisper_model, device)
        for job in todo:
            t0 = time.perf_counter()
            audio = media.load_audio(job.input, ASR_SAMPLE_RATE)
            raw = asr.transcribe(audio, language="en")
            segs = refine_boundaries(raw, audio)
            job.segments = merge_into_sentences(segs, cfg.merge_max_gap, cfg.merge_max_duration)
            job.timings["asr_s"] = round(time.perf_counter() - t0, 2)
            log.info("[%s] ASR: %d chunks -> %d segments", job.stem, len(raw), len(job.segments))
        asr = None
        _release_memory()

    # 2) Translation ---------------------------------------------------------
    if todo:
        log.info("Loading NLLB-200 (%s)...", cfg.nllb_model)
        mt = NLLBTranslator(cfg.nllb_model, device)
        for job in todo:
            t0 = time.perf_counter()
            translate_segments(job.segments, mt)
            job.timings["mt_s"] = round(time.perf_counter() - t0, 2)
            for s in job.segments:
                log.info("[%s] %.2f-%.2f  EN: %s\n%s PL: %s", job.stem, s.start, s.end, s.text_en, " " * 20, s.text_pl)
        mt = None
        _release_memory()

    # 3) Speech synthesis (voice cloning) ------------------------------------
    for job in jobs:
        if cfg.voice_sample is not None:
            job.voice_sample = Path(cfg.voice_sample)
        elif (job.work_dir / "voice_sample.wav").exists() and job not in todo:
            job.voice_sample = job.work_dir / "voice_sample.wav"
        else:
            job.voice_sample = auto_voice_sample(job, cfg.voice_sample_seconds)
    if todo:
        log.info("Loading XTTS-v2...")
        synth = None
        current_voice = None
        for job in todo:
            if synth is None or job.voice_sample != current_voice:
                if synth is None:
                    synth = XTTSSynthesizer(str(job.voice_sample), device, cfg.tts_model)
                else:  # same model, new speaker
                    synth.gpt_cond_latent, synth.speaker_embedding = synth.model.get_conditioning_latents(
                        audio_path=[str(job.voice_sample)]
                    )
                current_voice = job.voice_sample
            t0 = time.perf_counter()
            for i, seg in enumerate(job.segments):
                wav = synth.synthesize(seg.text_pl, "pl", seed=cfg.seed + i)
                out = job.work_dir / "tts" / f"seg_{i:03d}.wav"
                media.save_wav(out, wav, synth.sample_rate)
                seg.tts_path = str(out)
                seg.tts_duration = round(len(wav) / synth.sample_rate, 3)
            job.timings["tts_s"] = round(time.perf_counter() - t0, 2)
            save_segments(job.segments, job.segments_path)
        synth = None
        _release_memory()

    # 4) Synchronisation + remux, one output per method -----------------------
    for job in jobs:
        save_segments(job.segments, job.segments_path)
        write_srt(job.segments, job.work_dir / f"{job.stem}.en.srt", "en")
        write_srt(job.segments, job.work_dir / f"{job.stem}.pl.srt", "pl")
        if not job.segments:
            log.warning("[%s] no speech found, nothing to dub", job.stem)
            continue
        sr = None
        clips = []
        for seg in job.segments:
            with wave.open(seg.tts_path, "rb") as wf:
                sr = wf.getframerate()
            clips.append(media.load_audio(seg.tts_path, sr))
        slots = [(s.start, s.end) for s in job.segments]
        for method in cfg.methods:
            t0 = time.perf_counter()
            track, placements = build_track(
                method, clips, slots, sr, job.duration, cfg.min_stretch, cfg.max_stretch
            )
            wav_path = job.work_dir / f"{job.stem}.{method}.wav"
            media.save_wav(wav_path, track, sr)
            out_path = wav_path
            if job.is_video:
                out_path = job.work_dir / f"{job.stem}.{method}.mp4"
                media.remux(job.input, wav_path, out_path, extend_video_to=len(track) / sr)
            job.timings[f"sync_{method}_s"] = round(time.perf_counter() - t0, 2)
            job.outputs[method] = str(out_path)
            job.metrics[method] = timing_metrics(job.segments, placements, job.duration)
            log.info("[%s] %s -> %s", job.stem, method.upper(), out_path)

        report_path = job.work_dir / "report.json"
        if cfg.reuse and report_path.exists():
            # keep results of methods that were not re-rendered this time
            old = json.loads(report_path.read_text(encoding="utf-8"))
            job.outputs = {**old.get("outputs", {}), **job.outputs}
            job.metrics = {**old.get("metrics", {}), **job.metrics}
            job.timings = {**old.get("timings", {}), **job.timings}
        report = {
            "input": str(job.input),
            "duration_s": job.duration,
            "voice_sample": str(job.voice_sample),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
            "device": device,
            "timings": job.timings,
            "outputs": job.outputs,
            "metrics": job.metrics,
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return jobs

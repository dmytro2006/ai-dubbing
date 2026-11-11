"""Command-line interface: ``ai-dubbing INPUT [INPUT ...] [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .sync import METHODS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-dubbing",
        description="Dub English video into Polish (Whisper → NLLB-200 → XTTS-v2) "
        "with timing method M0 (no correction) and/or M1 (time-stretching).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path, help="input video or audio file(s)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"),
                   help="results go to OUTPUT_DIR/<input name>/")
    p.add_argument("-m", "--method", dest="methods", nargs="+", default=list(METHODS),
                   choices=[*METHODS, "all"], help="timing method(s) to render")
    p.add_argument("-v", "--voice", type=Path, default=None,
                   help="voice sample (WAV, ~6-15 s) to clone; default: cut from the input itself")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps ...")
    p.add_argument("--max-stretch", type=float, default=None,
                   help="M1: maximum speed-up factor (e.g. 1.3); default: no limit")
    p.add_argument("--min-stretch", type=float, default=None,
                   help="M1: maximum slow-down factor (e.g. 0.8); default: no limit")
    p.add_argument("--seed", type=int, default=0, help="random seed for XTTS sampling")
    p.add_argument("--reuse", action="store_true",
                   help="reuse transcription/translation/speech already in the output dir "
                        "(e.g. to re-render M1 with other limits)")
    p.add_argument("--whisper-model", default=None, help="Hugging Face id of the Whisper model")
    p.add_argument("--nllb-model", default=None, help="Hugging Face id of the NLLB model")
    p.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Third-party libraries log at WARNING; only our own logger is chattier.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("ai_dubbing").setLevel(logging.WARNING if args.quiet else logging.INFO)

    from .pipeline import DubbingConfig, run  # heavy imports only after parsing

    methods = tuple(METHODS) if "all" in args.methods else tuple(dict.fromkeys(args.methods))
    cfg = DubbingConfig(
        methods=methods,
        output_dir=args.output_dir,
        voice_sample=args.voice,
        device=args.device,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        seed=args.seed,
        reuse=args.reuse,
    )
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.nllb_model:
        cfg.nllb_model = args.nllb_model

    try:
        jobs = run(args.inputs, cfg)
    except (FileNotFoundError, RuntimeError) as e:
        logging.getLogger("ai_dubbing").error("%s", e)
        return 1

    for job in jobs:
        print(f"\n{job.input.name}  ({len(job.segments)} segments)")
        for method, path in job.outputs.items():
            s = job.metrics[method]["summary"]
            print(
                f"  {method.upper()}: {path}\n"
                f"      total duration error {s['total_rel_duration_error']:+.1%}, "
                f"mean |end error| {s['mean_abs_end_error_s']:.2f} s, "
                f"stretch k {s['min_stretch_factor']:.2f}-{s['max_stretch_factor']:.2f}"
            )
        print(f"  report: {job.work_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

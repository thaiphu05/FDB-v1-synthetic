#!/usr/bin/env python3
"""Stage 5 — check the dataset against the paper's constraints and report stats.

Checks per sample (failures are listed in the report, nothing is modified):
  all            16 kHz mono input.wav, meta.json present
  pause          pause in [0.4, 1.0] s, silent inside, speech on both sides,
                 turn >= min_pause_turn_sec
  turn-taking    silent after the [TURN-TAKING] start, tail >= 5 s, turn >= 4 s
  interruption   interrupt starts context + 7 s, 15 s tail, silent in between
  backchannel    GT present, window count = int(dur / 0.2) + 1, sums to 1

Writes out_dir/report.md.

Usage: python s5_check.py [--config config.yaml]
"""

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from common import TASK_DIRS, TASKS, load_config, stage_args

SILENT = 1e-4  # max |x| inside a region that must be silence
TOL = 0.01     # seconds


def load(path: Path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, sr


def region(x: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    return x[int(round(t0 * sr)) : int(round(t1 * sr))]


def check_sample(cfg, task: str, d: Path, gt: Dict[str, Any]) -> List[str]:
    p, errs = cfg["paper"], []
    x, sr = load(d / "input.wav")
    if sr != cfg["audio"]["sample_rate"] or x.shape[1] != 1:
        errs.append(f"input.wav {sr} Hz {x.shape[1]} ch")
    x = x[:, 0]
    dur = x.size / sr
    if not (d / "meta.json").is_file():
        errs.append("no meta.json")
    if task == "pause_handling":
        t0, t1 = json.loads((d / "pause.json").read_text())[0]["timestamp"]
        lo, hi = p["pause_sec"]
        if not lo - TOL <= t1 - t0 <= hi + TOL:
            errs.append(f"pause {t1 - t0:.3f}s")
        if np.abs(region(x, sr, t0, t1)).max(initial=0) > SILENT:
            errs.append("pause not silent")
        if np.abs(region(x, sr, 0, t0)).max(initial=0) <= SILENT or np.abs(region(x, sr, t1, dur)).max(initial=0) <= SILENT:
            errs.append("no speech around pause")
        if dur < p["min_pause_turn_sec"] - TOL:
            errs.append(f"turn {dur:.2f}s")
    elif task == "smooth_turn_taking":
        t0 = json.loads((d / "turn_taking.json").read_text())[0]["timestamp"][0]
        if np.abs(region(x, sr, t0, dur)).max(initial=0) > SILENT:
            errs.append("speech after turn end")
        if dur - t0 < p["turn_taking_tail_sec"] - TOL:
            errs.append(f"tail {dur - t0:.2f}s")
        if t0 < p["min_turn_taking_sec"] - TOL:
            errs.append(f"turn {t0:.2f}s")
    elif task == "user_interruption":
        t0, t1 = json.loads((d / "interrupt.json").read_text())[0]["timestamp"]
        ctx = load(d / "context.wav")[0].shape[0] / sr
        if abs(t0 - ctx - p["interrupt_gap_sec"]) > TOL:
            errs.append(f"gap {t0 - ctx:.2f}s")
        if abs(dur - t1 - p["interrupt_tail_sec"]) > TOL:
            errs.append(f"tail {dur - t1:.2f}s")
        if np.abs(region(x, sr, ctx, t0)).max(initial=0) > SILENT:
            errs.append("gap not silent")
    elif task == "backchannel":
        q = gt.get(d.name)
        if q is None:
            errs.append("no GT")
        else:
            n = int(dur / p["backchannel_window_sec"]) + 1
            if len(q) != n:
                errs.append(f"GT {len(q)} windows, expected {n}")
            if abs(sum(q) - 1) > 1e-6:
                errs.append(f"GT sum {sum(q):.4f}")
    return errs


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    out_dir: Path = cfg["paths"]["out_dir"]
    lines = ["# fdb-v1-gen report", ""]
    lines += ["| Task | Samples | Target | Duration mean (min–max) s | Topics | Voices | Errors |", "|---|---|---|---|---|---|---|"]
    details, all_errs = [], 0
    for task in TASKS:
        tdir = out_dir / TASK_DIRS[task]
        gt_path = tdir / "icc_gt_distribution.json"
        gt = json.loads(gt_path.read_text()) if gt_path.is_file() else {}
        dirs = sorted((d for d in tdir.glob("*") if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        durs, topics, voices, extra, n_err = [], Counter(), Counter(), Counter(), 0
        for d in dirs:
            errs = check_sample(cfg, task, d, gt)
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
            durs.append(meta.get("duration_sec", 0.0))
            topics[meta.get("source", {}).get("topic", "?")] += 1
            voices[meta.get("voice", "?")] += 1
            if task == "pause_handling":
                extra[meta.get("pause_by", "?")] += 1
            if errs:
                n_err += 1
                details.append(f"- `{TASK_DIRS[task]}/{d.name}`: {'; '.join(errs)}")
        all_errs += n_err
        dur = f"{mean(durs):.1f} ({min(durs):.1f}–{max(durs):.1f})" if durs else "-"
        lines.append(
            f"| {task} | {len(dirs)} | {cfg['counts'][task]} | {dur} | {len(topics)} | {len(voices)} | {n_err} |"
        )
        if extra:
            details.insert(0, f"- pause position: {dict(extra)}")
        print(f"[s5] {task}: {len(dirs)} samples, {n_err} with errors")
    skipped = out_dir / "skipped.jsonl"
    n_skip = sum(1 for _ in open(skipped)) if skipped.is_file() else 0
    lines += ["", f"Skipped in s4: {n_skip} (see skipped.jsonl)", "", "## Details", ""]
    lines += details or ["- no errors"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[s5] {all_errs} samples with errors -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

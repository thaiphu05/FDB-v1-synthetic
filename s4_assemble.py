#!/usr/bin/env python3
"""Stage 4 — rendered units + alignment -> FDB v1 dataset folders.

For each sample the unit wavs of a part are trimmed at their edges, joined
with a short sentence gap, resampled to 16 kHz mono, and laid out with the
paper's timing (Sec. III-C):

  synthetic_pause_handling/{ID}/       input.wav = a + pause + b, pause.json
  synthetic_smooth_turn_taking/{ID}/   input.wav = turn + 5 s, turn_taking.json
  synthetic_user_interruption/{ID}/    input.wav = context + 7 s + interrupt + 15 s,
                                       context.wav, interrupt.wav, interrupt.json
  synthetic_backchannel/{ID}/          input.wav = turn + 2 s
  synthetic_backchannel/icc_gt_distribution.json   {ID: Q per 200 ms window}

Event timestamps come from sample counts; word timestamps (transcription.json,
backchannel GT) from the s3 alignment. {ID} = sample number - 1, so IDs stay
stable when a sample is skipped. Skipped samples go to out_dir/skipped.jsonl.

Usage: python s4_assemble.py [--config config.yaml]
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common import (
    TASK_DIRS,
    batch_dir,
    even_words,
    load_config,
    load_mono,
    read_align_txt,
    read_jsonl,
    sample_num,
    stage_args,
    token_end_times,
    trim_bounds,
    write_json,
    write_jsonl,
    write_wav,
)


class Skip(Exception):
    pass


class Builder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = int(cfg["audio"]["sample_rate"])
        self.unit_dir: Path = cfg["paths"]["unit_audio_dir"]
        self.texts = {u["id"]: u["text"] for u in read_jsonl(batch_dir(cfg) / "omnivoice_units_fdb.jsonl")}
        fb = self.unit_dir / "align_fallback.txt"
        self.fallback = set(fb.read_text().split()) if fb.is_file() else set()

    def silence(self, sec: float) -> np.ndarray:
        return np.zeros(int(round(sec * self.sr)), dtype=np.float32)

    def unit(self, uid: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Trimmed unit audio and its words (times relative to the trimmed start)."""
        wav_path = self.unit_dir / f"{uid}.wav"
        if not wav_path.is_file():
            raise Skip(f"missing {wav_path.name}")
        a = self.cfg["audio"]
        x = load_mono(wav_path, self.sr)
        s, e = trim_bounds(x, self.sr, a["trim_db"], a["trim_margin_sec"])
        if e <= s:
            raise Skip(f"silent unit {uid}")
        seg = x[s:e]
        txt = wav_path.with_suffix(".txt")
        raw = read_align_txt(txt) if txt.is_file() else even_words(self.texts[uid], 0.0, x.size / self.sr)
        off, dur = s / self.sr, seg.size / self.sr
        ws = [
            {"word": w["word"], "start": min(max(w["start"] - off, 0.0), dur), "end": min(max(w["end"] - off, 0.0), dur)}
            for w in raw
        ]
        return seg, ws

    def part(self, uids: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Join the sentences of one part with a short gap."""
        gap = self.silence(self.cfg["audio"]["sentence_gap_sec"])
        pieces, ws, pos = [], [], 0
        for i, uid in enumerate(uids):
            if i:
                pieces.append(gap)
                pos += gap.size
            seg, uw = self.unit(uid)
            t0 = pos / self.sr
            ws += [{**w, "start": w["start"] + t0, "end": w["end"] + t0} for w in uw]
            pieces.append(seg)
            pos += seg.size
        return np.concatenate(pieces), ws


def shift(ws: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
    return [{**w, "start": w["start"] + t, "end": w["end"] + t} for w in ws]


def transcription(ws: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FDB example format: speaker 1 = the user stream in input.wav."""
    return [
        {
            "speaker": 1,
            "item": [{"text": w["word"], "timestamp": [round(w["start"], 4), round(w["end"], 4)]} for w in ws],
        }
    ]


def backchannel_gt(cfg, tokens, probs, ws, dur: float) -> List[float]:
    """Spread P(bc after word i) onto 200 ms windows at each word's end time."""
    p = cfg["paper"]
    win = p["backchannel_window_sec"]
    n = int(dur / win) + 1  # same window count eval_backchannel builds
    q = np.zeros(n)
    for t, prob in zip(token_end_times(tokens, ws), probs):
        q[min(int(t / win), n - 1)] += prob
    r = int(p["backchannel_gt_smooth"])
    if r > 0:
        kernel = np.concatenate([np.arange(1, r + 2), np.arange(r, 0, -1)]).astype(float)
        q = np.convolve(q, kernel / kernel.sum(), mode="same")
    q = q + 1e-12
    return (q / q.sum()).tolist()


def assemble(b: Builder, s: Dict[str, Any]) -> Dict[str, Any]:
    """Build one sample; returns {files: {name: array|json}, gt?: [...]}."""
    cfg, task, p = b.cfg, s["task"], b.cfg["paper"]
    parts = [b.part(ids) for ids in s["parts"]]
    out: Dict[str, Any] = {}
    if task == "pause_handling":
        (a, wa), (bb, wb) = parts
        pause = b.silence(s["pause_sec"])
        x = np.concatenate([a, pause, bb])
        if x.size / b.sr < p["min_pause_turn_sec"]:
            raise Skip(f"turn {x.size / b.sr:.2f}s < {p['min_pause_turn_sec']}s")
        a_end = a.size / b.sr
        out["pause.json"] = [{"text": "[PAUSE]", "timestamp": [round(a_end, 4), round(a_end + pause.size / b.sr, 4)]}]
        out["transcription.json"] = transcription(wa + shift(wb, (a.size + pause.size) / b.sr))
    elif task == "smooth_turn_taking":
        (turn, ws), = parts
        end = turn.size / b.sr
        if end < p["min_turn_taking_sec"]:
            raise Skip(f"turn {end:.2f}s < {p['min_turn_taking_sec']}s")
        x = np.concatenate([turn, b.silence(p["turn_taking_tail_sec"])])
        out["turn_taking.json"] = [
            {"text": "[TURN-TAKING]", "timestamp": [round(end, 4), round(end + p["turn_taking_window_sec"], 4)]}
        ]
        out["transcription.json"] = transcription(ws)
    elif task == "user_interruption":
        (ctx, _), (intr, _) = parts
        gap = b.silence(p["interrupt_gap_sec"])
        x = np.concatenate([ctx, gap, intr, b.silence(p["interrupt_tail_sec"])])
        i0 = (ctx.size + gap.size) / b.sr
        out["interrupt.json"] = [
            {
                "context": s["context"],
                "interrupt": s["interrupt"],
                "timestamp": [round(i0, 4), round(i0 + intr.size / b.sr, 4)],
            }
        ]
        out["context.wav"], out["interrupt.wav"] = ctx, intr
    elif task == "backchannel":
        (turn, ws), = parts
        bad = [u for u in s["parts"][0] if u in b.fallback or not (b.unit_dir / f"{u}.txt").is_file()]
        if bad:
            raise Skip(f"no alignment for {bad}")
        x = np.concatenate([turn, b.silence(p["backchannel_tail_sec"])])
        out["transcription.json"] = transcription(ws)
        out["_gt"] = backchannel_gt(cfg, s["words"], s["bc_probs"], ws, x.size / b.sr)
    else:
        raise ValueError(task)
    # one gain for every wav of the sample, so context/interrupt match input.wav
    peak = float(np.abs(x).max()) or 1.0
    gain = cfg["audio"]["peak"] / peak
    out["input.wav"] = x
    for k in list(out):
        if k.endswith(".wav"):
            out[k] = out[k] * gain
    return out


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    b = Builder(cfg)
    out_dir: Path = cfg["paths"]["out_dir"]
    samples = read_jsonl(batch_dir(cfg) / "samples.jsonl")
    gt: Dict[str, List[float]] = {}
    skipped, done = [], {}
    for s in samples:
        task = s["task"]
        sid = str(sample_num(s["sample_id"]) - 1)
        try:
            files = assemble(b, s)
        except Skip as exc:
            skipped.append({"sample_id": s["sample_id"], "reason": str(exc)})
            continue
        d = out_dir / TASK_DIRS[task] / sid
        for name, val in files.items():
            if name == "_gt":
                gt[sid] = val
            elif name.endswith(".wav"):
                write_wav(d / name, val, b.sr)
            else:
                write_json(d / name, val)
        meta = {k: v for k, v in s.items() if k not in ("parts", "bc_probs", "words")}
        write_json(d / "meta.json", {**meta, "units": s["parts"], "duration_sec": round(files["input.wav"].size / b.sr, 3)})
        done[task] = done.get(task, 0) + 1
    if gt:
        write_json(out_dir / TASK_DIRS["backchannel"] / "icc_gt_distribution.json", gt)
    write_jsonl(out_dir / "skipped.jsonl", skipped)
    for task, n in done.items():
        print(f"[s4] {TASK_DIRS[task]}: {n}")
    print(f"[s4] skipped {len(skipped)} -> {out_dir / 'skipped.jsonl'}")
    for r in skipped[:10]:
        print(f"  {r['sample_id']}: {r['reason']}")


if __name__ == "__main__":
    main()

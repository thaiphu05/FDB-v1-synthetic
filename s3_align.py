#!/usr/bin/env python3
"""Stage 3 — forced-align every rendered unit: {id}.wav -> {id}.txt.

Same aligner as vilex (Qwen3-ForcedAligner via transformers), written here
standalone so fdb-v1-gen does not import ~/Vilex. Output is the vilex align
dump format, one word per line: ``<start> <end> <word>`` (seconds, relative to
the start of the unit wav).

A unit the aligner cannot handle gets an evenly-split fallback .txt and its id
is listed in align_fallback.txt (s4 drops backchannel samples that use one).
Units that already have a .txt are skipped unless --force.

Runs on B200 (aligner.device: cuda) or locally (cpu).

Usage: python s3_align.py [--config config.yaml] [--force]
"""

from pathlib import Path
from typing import Any, Dict, List

from common import (
    batch_dir,
    even_words,
    load_config,
    load_mono,
    read_jsonl,
    stage_args,
    write_align_txt,
)

ALIGN_SR = 16000  # feature extractor rate


def resolve_dtype(torch, name: str, device: str):
    """auto -> bf16 on Ampere+, fp16 on older GPUs, fp32 on cpu (as vilex)."""
    if name in ("auto", "", None):
        if device.startswith("cuda") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[str(name).lower()]


class Aligner:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        a = cfg["aligner"]
        device = a["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[s3] cuda not available, using cpu")
            device = "cpu"
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(a["model"])
        self.model = AutoModelForTokenClassification.from_pretrained(
            a["model"], dtype=resolve_dtype(torch, a["dtype"], device)
        ).to(device)
        self.model.eval()
        print(f"[s3] aligner {a['model']} on {device} ({self.model.dtype})")

    def align(self, wavs: List[Any], texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Word timestamps per (wav @16 kHz, text) pair; raises on model errors."""
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=wavs, transcript=texts, language=None
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        stamps = self.processor.decode_forced_alignment(
            logits=logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )
        return [
            [
                {"word": it.get("text", ""), "start": float(it["start_time"]), "end": float(it["end_time"])}
                for it in items
                if "start_time" in it and "end_time" in it
            ]
            for items in stamps
        ]


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    unit_dir: Path = cfg["paths"]["unit_audio_dir"]
    units = read_jsonl(batch_dir(cfg) / "omnivoice_units_fdb.jsonl")
    todo, missing = [], 0
    for u in units:
        wav = unit_dir / f"{u['id']}.wav"
        if not wav.is_file():
            missing += 1
        elif args.force or not wav.with_suffix(".txt").is_file():
            todo.append(u)
    print(f"[s3] {len(units)} units, {missing} wav missing, {len(todo)} to align")
    if not todo:
        return

    aligner = Aligner(cfg)
    bs = int(cfg["aligner"]["batch_size"])
    fallback_path = unit_dir / "align_fallback.txt"
    fallback = set(fallback_path.read_text().split()) if fallback_path.is_file() else set()
    for i in range(0, len(todo), bs):
        chunk = todo[i : i + bs]
        wavs = [load_mono(unit_dir / f"{u['id']}.wav", ALIGN_SR) for u in chunk]
        try:
            results = aligner.align(wavs, [u["text"] for u in chunk])
        except Exception:  # retry one by one so a bad unit does not sink the batch
            results = []
            for u, w in zip(chunk, wavs):
                try:
                    results.append(aligner.align([w], [u["text"]])[0])
                except Exception as exc:
                    print(f"  align failed {u['id']}: {exc}")
                    results.append([])
        for u, w, ws in zip(chunk, wavs, results):
            if ws:
                fallback.discard(u["id"])
            else:
                ws = even_words(u["text"], 0.0, w.size / ALIGN_SR)
                fallback.add(u["id"])
            write_align_txt(unit_dir / f"{u['id']}.txt", ws)
        print(f"  {min(i + bs, len(todo))}/{len(todo)}")
    fallback_path.write_text("".join(f"{x}\n" for x in sorted(fallback)))
    print(f"[s3] done, {len(fallback)} fallback units -> {fallback_path}")


if __name__ == "__main__":
    main()

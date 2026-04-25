#!/usr/bin/env python3
"""Annotate NuScenes keyframes with normal_core / not_normal_core / uncertain.

Standalone script. Does NOT depend on extract.py or pipeline.py.

Pipeline:
  1. One-time extract v1.0-trainval_meta.tgz to get sample.json / sample_data.json.
  2. Build sample_token -> {6 camera jpg paths} from sample_data.json, keeping
     only keyframe samples whose 6 CAM_* jpgs all exist under --samples-root.
  3. Load a VLM (default Cosmos-Reason1-7B, optionally Qwen3.5-9B) in bf16 on
     a single CUDA device. Processor is configured with max_pixels large enough
     that 1600x900 NuScenes frames are not downscaled.
  4. For each sample, feed all 6 cameras (in the canonical order) + a prompt
     that contains the full normal_core / not_normal_core / uncertain rubric
     (derived from latest_grading_criteria.md). The model is asked to produce
     a short analysis followed by "Final answer: X" where X in {A, B, C}.
  5. generate() is run with output_scores=True. We locate the generated token
     corresponding to the final letter, take its pre-softmax logits, restrict
     to the three letter token ids, renormalize. That gives per-class
     probabilities; label = argmax, confidence = max prob.
  6. Results are written incrementally (atomic rewrite every --save-every
     samples) to a single JSON array. --resume skips sample_tokens already
     present in the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

LETTER_TO_LABEL = {
    "A": "normal_core",
    "B": "not_normal_core",
    "C": "uncertain",
}
LABELS_IN_ORDER = ["normal_core", "not_normal_core", "uncertain"]

# Instruction built from dda4080_huawei_project/latest_grading_criteria.md.
# Unlike the representation-extraction prompt (§Prompt Principle), an annotator
# *needs* the explicit definition to produce a classification, so we include it.
PROMPT_TEMPLATE = """You are given 6 synchronized camera images from an autonomous vehicle at a single timestamp. The cameras are provided in this order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

Your task: assign exactly ONE label to this multi-camera frame.

Definition - normal_core:
A simple driving scene where, based on the visible static and weakly-dynamic information, the ego vehicle does NOT need to take any defensive action beyond routine driving. Concretely, a sample is normal_core only if ALL of the following hold:
- The main road and the relevant traffic participants are clearly visible.
- Lighting and weather do not significantly degrade perception (no night, no strong glare, no rain, no wet reflective surfaces, no fog, no low visibility).
- The ego vehicle has a reasonable safe buffer: no close lead vehicle, no visible brake lights of the lead vehicle, no congestion ahead, no sign of impending deceleration.
- The road ahead is clear: no construction zone, no cones, no barriers, no closure, no abnormally parked vehicles, no lane-intruding obstacles.
- Pedestrians, cyclists, and other vulnerable road users are in routine positions, not near the ego path, and not occluded in a way that might hide risk.
- Large vehicles or buildings do not severely occlude key risk areas.
- The road structure is clear (no unusually complex intersection, no ambiguous lane boundaries, no abnormal topology).
- No camera is severely blurred, occluded, over-exposed, under-exposed, or otherwise abnormal.
- Nothing visible requires extra deceleration, yielding, increased following distance, or other conservative behavior.

If the visible information clearly indicates ANY of the exclusion conditions above (degraded perception conditions, close lead / braking / congestion, VRUs near ego path or occluded, construction / obstacles / abnormal parking, complex or ambiguous road structure, abnormal image quality, or anything requiring extra defensive driving), the sample is NOT normal_core.

If a single-frame multi-camera view is genuinely insufficient to make a stable judgement (strongly ambiguous even after careful inspection), choose uncertain. Do NOT use uncertain as a hedge when the answer is actually clear.

Important: do NOT infer temporal events (no cut-in, no sudden braking, no jaywalking across time). Reason only from what is statically visible in this single multi-camera frame.

Output format (strict):
1. First, write 2 to 4 short sentences analysing the visible static driving-relevant factors (lighting, weather, visibility, road layout, traffic participants, obstacles, construction, image quality).
2. Then, on a new line, output exactly:
Final answer: X
where X is one of:
A (normal_core)       - high-confidence normal scene
B (not_normal_core)   - clearly requires extra defensive driving, or has static difficult factors
C (uncertain)         - single-frame information is genuinely insufficient
"""


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    proj = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-root", type=Path,
                   default=proj / "dataset" / "nuscenes" / "nuscenes_extracted" / "samples",
                   help="Directory containing CAM_FRONT/, CAM_FRONT_LEFT/, ... subdirs with jpgs.")
    p.add_argument("--meta-tgz", type=Path,
                   default=proj / "dataset" / "nuscenes" / "v1.0-trainval_meta.tgz")
    p.add_argument("--meta-extract-dir", type=Path,
                   default=proj / "dataset" / "nuscenes",
                   help="Parent dir where v1.0-trainval/ metadata will be extracted.")
    p.add_argument("--model-dir", type=Path,
                   default=proj / "models" / "Cosmos-Reason1-7B",
                   help="HF-format model directory. Switch to models/Qwen3.5-9B to try Qwen.")
    p.add_argument("--output", type=Path,
                   default=proj / "output" / "annotations_output" / "normal_core_annotations.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-pixels", type=int, default=1_600_000,
                   help="NuScenes frames are 1600x900 = 1_440_000 px. "
                        "Keep this >= 1_440_000 to preserve original resolution.")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--enable-thinking", action="store_true", default=False,
                   help="If set, allow chain-of-thought (Qwen3.5 default). Off by default "
                        "to keep generation short and force a direct 'Final answer: X'.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 means annotate all eligible samples.")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Enabled by default. Pass --no-resume to start fresh.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Metadata: extract meta.tgz and group keyframes by sample_token
# ---------------------------------------------------------------------------

def ensure_meta(meta_tgz: Path, extract_dir: Path) -> Path:
    """Extract v1.0-trainval/ once and return the directory containing the JSONs."""
    target = extract_dir / "v1.0-trainval"
    marker = extract_dir / ".v1.0-trainval_meta.done"
    if target.exists() and marker.exists():
        return target
    if not meta_tgz.exists():
        raise FileNotFoundError(f"meta tgz not found: {meta_tgz}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[meta] extracting {meta_tgz.name} -> {extract_dir}")
    subprocess.run(["tar", "-xzf", str(meta_tgz), "-C", str(extract_dir)], check=True)
    if not target.exists():
        raise RuntimeError(f"extraction finished but {target} is missing")
    marker.write_text("ok")
    return target


def build_samples(meta_dir: Path, samples_root: Path) -> list[dict]:
    """Return sorted list of {sample_token, channels: {CAM_*: Path}} for keyframes
    whose all 6 CAM_* jpgs exist on disk under samples_root."""
    sd_path = meta_dir / "sample_data.json"
    s_path = meta_dir / "sample.json"
    if not sd_path.exists() or not s_path.exists():
        raise FileNotFoundError(f"missing metadata under {meta_dir}")

    print(f"[meta] loading {sd_path.name} and {s_path.name}")
    with sd_path.open() as f:
        sample_data = json.load(f)
    with s_path.open() as f:
        samples_meta = json.load(f)
    sample_ts = {s["token"]: s.get("timestamp", 0) for s in samples_meta}

    # Group by sample_token. filename is like "samples/CAM_FRONT/xxx.jpg".
    by_sample: dict[str, dict[str, Path]] = {}
    for sd in sample_data:
        if not sd.get("is_key_frame"):
            continue
        fname = sd.get("filename", "")
        if not fname.startswith("samples/"):
            continue
        parts = fname.split("/")
        if len(parts) < 3:
            continue
        channel = parts[1]
        if channel not in CAMERAS:
            continue
        # map samples/CAM_X/basename.jpg -> samples_root/CAM_X/basename.jpg
        local_path = samples_root / channel / parts[2]
        by_sample.setdefault(sd["sample_token"], {})[channel] = local_path

    eligible: list[dict] = []
    for token, channels in by_sample.items():
        if any(cam not in channels for cam in CAMERAS):
            continue
        if not all(channels[cam].exists() for cam in CAMERAS):
            continue
        eligible.append({
            "sample_token": token,
            "timestamp": sample_ts.get(token, 0),
            "channels": {cam: channels[cam] for cam in CAMERAS},
        })

    eligible.sort(key=lambda r: (r["timestamp"], r["sample_token"]))
    print(f"[meta] keyframes with all 6 cams on disk: {len(eligible)} "
          f"(total keyframe sample_tokens in meta: {len(by_sample)})")
    return eligible


# ---------------------------------------------------------------------------
#  Output IO (atomic rewrite) + resume
# ---------------------------------------------------------------------------

def load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    if not output_path.exists():
        return [], set()
    with output_path.open() as f:
        data = json.load(f)
    done = {row["sample_token"] for row in data if "sample_token" in row}
    print(f"[resume] loaded {len(data)} existing annotations from {output_path}")
    return data, done


def atomic_write_json(output_path: Path, data: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
#  Model loading and letter-token resolution
# ---------------------------------------------------------------------------

def load_vlm(model_dir: Path, device: str, dtype: str, min_pixels: int, max_pixels: int):
    import transformers
    from transformers import AutoProcessor

    print(f"[model] loading processor from {model_dir} "
          f"(min_pixels={min_pixels}, max_pixels={max_pixels})")
    processor = AutoProcessor.from_pretrained(
        str(model_dir), min_pixels=min_pixels, max_pixels=max_pixels,
    )

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    print(f"[model] loading weights from {model_dir} (dtype={dtype}, device={device})")
    t0 = time.time()

    # Try in order: AutoModelForMultimodalLM (transformers >=5, Qwen3.5),
    #               AutoModelForImageTextToText (transformers >=4.49, Qwen2.5-VL/Cosmos).
    auto_classes = []
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            auto_classes.append((name, cls))
    if not auto_classes:
        raise RuntimeError("transformers exposes neither AutoModelForMultimodalLM nor "
                           "AutoModelForImageTextToText; please upgrade transformers.")

    last_err = None
    for name, cls in auto_classes:
        try:
            print(f"[model] trying {name}")
            model = cls.from_pretrained(
                str(model_dir), dtype=torch_dtype, device_map=device,
            )
            print(f"[model] loaded with {name} ({type(model).__name__})")
            break
        except (ValueError, KeyError) as e:
            last_err = e
            print(f"[model] {name} could not load: {repr(e)[:200]}")
            model = None
    else:
        raise RuntimeError(
            f"None of the auto classes could load {model_dir}. Last error: {last_err}"
        )

    model.eval()
    print(f"[model] ready in {time.time() - t0:.1f}s")
    return processor, model


def resolve_letter_token_ids(tokenizer) -> dict[str, int]:
    """For each of A/B/C pick the token id that is most likely emitted by the
    model after a space (as in 'Final answer: A'). Fall back to the raw letter
    encoding if no single-token form exists."""
    ids: dict[str, int] = {}
    for letter in "ABC":
        spaced = tokenizer.encode(f" {letter}", add_special_tokens=False)
        raw = tokenizer.encode(letter, add_special_tokens=False)
        if len(spaced) == 1:
            ids[letter] = spaced[0]
        elif len(raw) == 1:
            ids[letter] = raw[0]
        else:
            # last resort: take the last sub-token
            ids[letter] = (spaced or raw)[-1]
    return ids


# ---------------------------------------------------------------------------
#  Per-sample inference
# ---------------------------------------------------------------------------

FINAL_ANSWER_RE = re.compile(r"Final answer:\s*([ABC])", re.IGNORECASE)


@torch.no_grad()
def annotate_one(model, processor, images: list[Image.Image],
                 letter_token_ids: dict[str, int],
                 max_new_tokens: int,
                 enable_thinking: bool = False) -> dict:
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": im} for im in images],
            {"type": "text", "text": PROMPT_TEMPLATE},
        ],
    }]
    # enable_thinking is Qwen3.x specific; older templates (Cosmos / Qwen2.5-VL)
    # ignore unknown kwargs.
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(
        text=[prompt], images=images, return_tensors="pt", padding=True,
    )
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0, input_len:].tolist()
    scores = out.scores  # tuple[len(gen_ids)] of Tensor[1, vocab]

    reasoning = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Locate the classification step. We require an explicit
    # "Final answer: X" pattern; otherwise we mark parse_ok=false rather than
    # picking up a stray A/B/C token from the middle of the reasoning.
    chosen_step: int | None = None
    match = FINAL_ANSWER_RE.search(reasoning)
    if match:
        letter_in_text = match.group(1).upper()
        for i in range(len(gen_ids) - 1, -1, -1):
            tok = processor.tokenizer.decode([gen_ids[i]]).strip()
            if tok.upper() == letter_in_text:
                chosen_step = i
                break

    if chosen_step is None:
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "per_class_probs": {lbl: 0.0 for lbl in LABELS_IN_ORDER},
            "reasoning": reasoning,
            "parse_ok": False,
        }

    step_logits = scores[chosen_step][0].float()
    letter_logits = torch.tensor(
        [step_logits[letter_token_ids[L]] for L in "ABC"],
        device=step_logits.device,
    )
    letter_probs = torch.softmax(letter_logits, dim=-1).tolist()

    per_class = {LETTER_TO_LABEL[L]: float(p)
                 for L, p in zip("ABC", letter_probs)}
    label = max(per_class, key=per_class.get)
    confidence = per_class[label]

    return {
        "label": label,
        "confidence": confidence,
        "per_class_probs": per_class,
        "reasoning": reasoning,
        "parse_ok": True,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    meta_dir = ensure_meta(args.meta_tgz, args.meta_extract_dir)
    samples = build_samples(meta_dir, args.samples_root)
    if not samples:
        print("[error] no eligible samples found on disk, abort.")
        sys.exit(1)

    results, done = load_existing(args.output) if args.resume else ([], set())
    pending = [s for s in samples if s["sample_token"] not in done]
    if args.max_samples > 0:
        pending = pending[: args.max_samples]
    print(f"[plan] total={len(samples)} done={len(done)} to_process={len(pending)}")
    if not pending:
        print("[plan] nothing to do.")
        return

    processor, model = load_vlm(
        args.model_dir, args.device, args.dtype, args.min_pixels, args.max_pixels,
    )
    letter_ids = resolve_letter_token_ids(processor.tokenizer)
    print(f"[model] letter token ids: {letter_ids}")

    n_ok = 0
    n_fail_parse = 0
    t_start = time.time()

    try:
        for i, sample in enumerate(pending, start=1):
            t0 = time.time()
            try:
                images = [Image.open(sample["channels"][cam]).convert("RGB")
                          for cam in CAMERAS]
            except Exception as e:
                print(f"[warn] skip {sample['sample_token']}: failed to open images ({e})")
                continue

            try:
                result = annotate_one(
                    model, processor, images, letter_ids, args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[oom] sample {sample['sample_token']} OOM, skipping. "
                      f"Consider lowering --max-pixels.")
                for im in images:
                    im.close()
                continue
            finally:
                for im in images:
                    im.close()

            record = {"sample_token": sample["sample_token"], **result}
            results.append(record)

            if result["parse_ok"]:
                n_ok += 1
            else:
                n_fail_parse += 1

            dt = time.time() - t0
            avg = (time.time() - t_start) / i
            eta_min = (len(pending) - i) * avg / 60
            if i == 1 or i % 5 == 0:
                print(f"[prog] {i}/{len(pending)} token={sample['sample_token'][:8]} "
                      f"label={record['label']} conf={record['confidence']:.3f} "
                      f"| {dt:.1f}s | avg {avg:.1f}s/it | ETA {eta_min:.0f}min "
                      f"| ok={n_ok} parse_fail={n_fail_parse}")

            if i % args.save_every == 0:
                atomic_write_json(args.output, results)
    finally:
        atomic_write_json(args.output, results)

    print(f"[done] wrote {len(results)} records to {args.output}")
    print(f"[done] this run: ok={n_ok} parse_fail={n_fail_parse} "
          f"elapsed={(time.time() - t_start) / 60:.1f}min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract Cosmos-Reason1 representations for selected NuScenes clips.

This script does exactly three things:
1) untar required keyframe archives (default: parts 04/05/06),
2) extract last-layer MLP output and last-layer MLP high-dim intermediate,
3) generate final text output for prompt: "describe the video in detail".

Outputs are written into three files under output directory:
  - last_mlp_output.npy
  - last_mlp_intermediate.npy
  - final_text_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    default_ann = data_dir / "vlm_annotated_clips_part04_06.json"
    if not default_ann.exists():
        default_ann = data_dir / "vlm_annotated_clips.json"

    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--tgz-dir", type=Path, default=data_dir)
    p.add_argument("--extract-root", type=Path, default=data_dir / "nuscenes_extracted")
    p.add_argument("--annotations", type=Path, default=default_ann)
    p.add_argument("--model-dir", type=Path, default=root / "models" / "Cosmos-Reason1-7B")
    p.add_argument("--output-dir", type=Path, default=root / "cosmos_extract_output")
    p.add_argument("--prompt", type=str, default="describe the video in detail")
    p.add_argument("--frames-per-camera", type=int, default=16)
    p.add_argument("--max-clips", type=int, default=0,
                   help="0 means all clips")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs in --output-dir if present",
    )
    return p.parse_args()


def select_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, num=k, dtype=int).tolist()


def extract_archives(parts: list[int], tgz_dir: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for part in parts:
        tgz_path = tgz_dir / f"v1.0-trainval{part:02d}_keyframes.tgz"

        marker = extract_root / f".part_{part:02d}.done"
        if marker.exists():
            print(f"[untar] part {part:02d} already extracted, skip")
            continue

        if not tgz_path.exists():
            samples_dir = extract_root / "samples"
            if samples_dir.exists() and any(samples_dir.iterdir()):
                print(f"[untar] part {part:02d} tgz missing but samples/ exists, skip")
                marker.write_text("pre-extracted")
                continue
            raise FileNotFoundError(f"Missing archive: {tgz_path}")

        marker_state = f"{tgz_path.stat().st_size}:{int(tgz_path.stat().st_mtime)}"
        print(f"[untar] extracting {tgz_path.name} -> {extract_root}")
        subprocess.run(
            ["tar", "-xzf", str(tgz_path), "-C", str(extract_root)],
            check=True,
        )
        marker.write_text(marker_state)


def channel_path_to_local(raw_path: str, extract_root: Path) -> Path | None:
    if not raw_path:
        return None
    p = raw_path.replace("\\", "/")
    idx = p.find("samples/")
    if idx < 0:
        return None
    rel = p[idx:]
    return extract_root / rel


def collect_images_for_camera(
    clip: dict,
    extract_root: Path,
    frames_per_camera: int,
    camera: str,
) -> list[Image.Image]:
    frames = clip.get("frames", [])
    indices = select_indices(len(frames), frames_per_camera)
    images: list[Image.Image] = []

    for fi in indices:
        channels = frames[fi].get("channels", {})
        raw = channels.get(camera)
        local_path = channel_path_to_local(raw, extract_root) if raw else None
        if local_path is None or not local_path.exists():
            continue
        images.append(Image.open(local_path).convert("RGB"))

    return images


def summarize_tokens(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3:
        raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    tokens = hidden[0]
    return tokens.mean(dim=0), tokens[-1]


def save_checkpoint(
    output_dir: Path,
    out_mean_arr: list[np.ndarray],
    out_last_arr: list[np.ndarray],
    mid_mean_arr: list[np.ndarray],
    mid_last_arr: list[np.ndarray],
    texts: list[dict],
    meta: list[dict],
) -> None:
    if not out_mean_arr:
        return
    np.save(output_dir / "last_mlp_output.npy", np.stack(out_mean_arr).astype(np.float32))
    np.save(output_dir / "last_mlp_output_last_token.npy", np.stack(out_last_arr).astype(np.float32))
    np.save(output_dir / "last_mlp_intermediate.npy", np.stack(mid_mean_arr).astype(np.float32))
    np.save(output_dir / "last_mlp_intermediate_last_token.npy", np.stack(mid_last_arr).astype(np.float32))
    with (output_dir / "final_text_output.jsonl").open("w", encoding="utf-8") as f:
        for row in texts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_checkpoint(
    output_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict], list[dict]]:
    out_mean_path = output_dir / "last_mlp_output.npy"
    out_last_path = output_dir / "last_mlp_output_last_token.npy"
    mid_mean_path = output_dir / "last_mlp_intermediate.npy"
    mid_last_path = output_dir / "last_mlp_intermediate_last_token.npy"
    text_path = output_dir / "final_text_output.jsonl"
    meta_path = output_dir / "meta.json"

    if not (
        out_mean_path.exists()
        and out_last_path.exists()
        and mid_mean_path.exists()
        and mid_last_path.exists()
        and text_path.exists()
        and meta_path.exists()
    ):
        return [], [], [], [], [], []

    out_mean_np = np.load(out_mean_path)
    out_last_np = np.load(out_last_path)
    mid_mean_np = np.load(mid_mean_path)
    mid_last_np = np.load(mid_last_path)
    with text_path.open("r", encoding="utf-8") as f:
        text_rows = [json.loads(line) for line in f if line.strip()]
    with meta_path.open("r", encoding="utf-8") as f:
        meta_rows = json.load(f)

    n = out_mean_np.shape[0]
    if (
        out_last_np.shape[0] != n
        or mid_mean_np.shape[0] != n
        or mid_last_np.shape[0] != n
        or len(text_rows) != n
        or len(meta_rows) != n
    ):
        raise ValueError(
            "Checkpoint row-count mismatch: "
            f"out_mean={n}, out_last={out_last_np.shape[0]}, "
            f"mid_mean={mid_mean_np.shape[0]}, mid_last={mid_last_np.shape[0]}, "
            f"text={len(text_rows)}, meta={len(meta_rows)}"
        )

    out_mean_rows = [out_mean_np[i] for i in range(n)]
    out_last_rows = [out_last_np[i] for i in range(n)]
    mid_mean_rows = [mid_mean_np[i] for i in range(n)]
    mid_last_rows = [mid_last_np[i] for i in range(n)]
    return out_mean_rows, out_last_rows, mid_mean_rows, mid_last_rows, text_rows, meta_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Step 1: images must be extracted before model inference.
    extract_archives(args.parts, args.tgz_dir, args.extract_root)

    with args.annotations.open("r", encoding="utf-8") as f:
        clips = json.load(f)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    print(f"[data] clips to process: {len(clips)} from {args.annotations}")

    print(f"[model] loading from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    captures: dict[str, torch.Tensor] = {}

    def mlp_out_hook(_module, _inputs, output):
        captures["mlp_out"] = output.detach()

    def down_proj_prehook(_module, inputs):
        captures["mlp_mid"] = inputs[0].detach()

    last_mlp = model.model.language_model.layers[-1].mlp
    h1 = last_mlp.register_forward_hook(mlp_out_hook)
    h2 = last_mlp.down_proj.register_forward_pre_hook(down_proj_prehook)

    out_dim = model.config.hidden_size
    mid_dim = last_mlp.down_proj.in_features

    mlp_out_mean_rows: list[np.ndarray] = []
    mlp_out_last_rows: list[np.ndarray] = []
    mlp_mid_mean_rows: list[np.ndarray] = []
    mlp_mid_last_rows: list[np.ndarray] = []
    text_rows: list[dict] = []
    meta_rows: list[dict] = []
    start_clip_idx = 0

    if args.resume:
        (
            mlp_out_mean_rows,
            mlp_out_last_rows,
            mlp_mid_mean_rows,
            mlp_mid_last_rows,
            text_rows,
            meta_rows,
        ) = load_checkpoint(args.output_dir)
        if meta_rows:
            # Resume from the next clip after the last saved global_clip_index.
            clip_pos_by_global_idx: dict[int, int] = {}
            for pos, c in enumerate(clips):
                gidx = c.get("global_clip_index")
                if isinstance(gidx, int):
                    clip_pos_by_global_idx[gidx] = pos

            last_global = meta_rows[-1].get("global_clip_index")
            if isinstance(last_global, int) and last_global in clip_pos_by_global_idx:
                start_clip_idx = clip_pos_by_global_idx[last_global] + 1
                print(
                    f"[resume] found {len(meta_rows)} saved rows, "
                    f"continuing from clip index {start_clip_idx}/{len(clips)}"
                )
            else:
                start_clip_idx = len(meta_rows)
                print(
                    "[resume] last global_clip_index not found in annotations; "
                    f"fallback start at clip index {start_clip_idx}/{len(clips)}"
                )
        else:
            print("[resume] no existing checkpoint rows found, starting from scratch")

    skipped = max(start_clip_idx - len(meta_rows), 0)

    try:
        for idx, clip in enumerate(clips[start_clip_idx:], start=start_clip_idx + 1):
            clip_out_mean = np.zeros((len(CAMERAS), out_dim), dtype=np.float32)
            clip_out_last = np.zeros((len(CAMERAS), out_dim), dtype=np.float32)
            clip_mid_mean = np.zeros((len(CAMERAS), mid_dim), dtype=np.float32)
            clip_mid_last = np.zeros((len(CAMERAS), mid_dim), dtype=np.float32)
            camera_texts: dict[str, str | None] = {}
            n_images_per_camera: dict[str, int] = {}
            valid_camera_count = 0

            for cam_i, cam in enumerate(CAMERAS):
                images = collect_images_for_camera(
                    clip, args.extract_root, args.frames_per_camera, cam
                )
                n_images_per_camera[cam] = len(images)
                if not images:
                    camera_texts[cam] = None
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": im} for im in images],
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[prompt],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                captures.clear()
                with torch.no_grad():
                    model.model(**inputs)
                if "mlp_out" not in captures or "mlp_mid" not in captures:
                    raise RuntimeError("Failed to capture MLP hooks on this sample")

                pooled_out_mean, pooled_out_last = summarize_tokens(captures["mlp_out"])
                pooled_mid_mean, pooled_mid_last = summarize_tokens(captures["mlp_mid"])
                # Captured tensors are full-sequence activations and can be very large.
                # Release them before text generation to avoid peak-memory spikes.
                captures.clear()

                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs["input_ids"], gen_ids, strict=False)
                ]
                text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                clip_out_mean[cam_i] = pooled_out_mean.float().cpu().numpy()
                clip_out_last[cam_i] = pooled_out_last.float().cpu().numpy()
                clip_mid_mean[cam_i] = pooled_mid_mean.float().cpu().numpy()
                clip_mid_last[cam_i] = pooled_mid_last.float().cpu().numpy()
                camera_texts[cam] = text
                valid_camera_count += 1

                for im in images:
                    im.close()

            if valid_camera_count == 0:
                skipped += 1
                continue

            mlp_out_mean_rows.append(clip_out_mean)
            mlp_out_last_rows.append(clip_out_last)
            mlp_mid_mean_rows.append(clip_mid_mean)
            mlp_mid_last_rows.append(clip_mid_last)
            text_rows.append(
                {
                    "global_clip_index": clip.get("global_clip_index"),
                    "scene_name": clip.get("scene_name"),
                    "tail_score": clip.get("tail_score"),
                    "prompt": args.prompt,
                    "camera_outputs": camera_texts,
                }
            )
            meta_rows.append(
                {
                    "row_index": len(meta_rows),
                    "global_clip_index": clip.get("global_clip_index"),
                    "scene_name": clip.get("scene_name"),
                    "tail_score": clip.get("tail_score"),
                    "frames_per_camera": args.frames_per_camera,
                    "n_images_per_camera": n_images_per_camera,
                    "valid_camera_count": valid_camera_count,
                }
            )

            if idx % args.save_every == 0:
                save_checkpoint(
                    args.output_dir,
                    mlp_out_mean_rows,
                    mlp_out_last_rows,
                    mlp_mid_mean_rows,
                    mlp_mid_last_rows,
                    text_rows,
                    meta_rows,
                )
                print(f"[progress] {idx}/{len(clips)} processed, kept={len(meta_rows)}, skipped={skipped}")

    finally:
        h1.remove()
        h2.remove()

    save_checkpoint(
        args.output_dir,
        mlp_out_mean_rows,
        mlp_out_last_rows,
        mlp_mid_mean_rows,
        mlp_mid_last_rows,
        text_rows,
        meta_rows,
    )
    print("[done] extraction finished")
    print(f"[done] kept={len(meta_rows)} skipped={skipped}")
    print(f"[done] file: {args.output_dir / 'last_mlp_output.npy'}")
    print(f"[done] file: {args.output_dir / 'last_mlp_output_last_token.npy'}")
    print(f"[done] file: {args.output_dir / 'last_mlp_intermediate.npy'}")
    print(f"[done] file: {args.output_dir / 'last_mlp_intermediate_last_token.npy'}")
    print(f"[done] file: {args.output_dir / 'final_text_output.jsonl'}")


if __name__ == "__main__":
    main()

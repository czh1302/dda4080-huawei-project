#!/usr/bin/env python3
"""Extract Cosmos-Reason1 decoder-layer MLP outputs for selected NuScenes clips.

This script does exactly three things:
1) untar required keyframe archives (default: parts 04/05/06),
2) extract selected decoder layers' MLP outputs,
3) generate final text output for prompt: "describe the video in detail".

For each selected layer L, two feature files are written:
    - layerLL_mlp_output_mean.npy       (all-token mean)
    - layerLL_mlp_output_last_token.npy (last token)

With default layers [15, 18, 21, 24, 27, 32], this produces 12 feature files.
By default the script uses all available frames per clip and feeds the six
camera views together as one multi-view sample for each frame.
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

DEFAULT_LAYERS_1BASED = [15, 18, 21, 24, 27, 32]
DEFAULT_PROMPT = (
    "These six images are synchronized views from the ego vehicle for one single frame, ordered as front, front-left, front-right, back, back-left, back-right. "
    "The image is taken from inside the ego vehicle looking out through the windshield onto a road and you are the driver of the ego vehicle. "
    "Please describe the driving condition including: overall scene type (urban, highway, residential, etc.), weather and lighting conditions, road layout and traffic conditions, dynamic objects (vehicles, pedestrians, cyclists), potential hazards or important interactions, and any risky situations or behaviors. "
    "Provide a concise description in one paragraph with less than 150 words."
)


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
    p.add_argument("--annotations", type=Path, nargs="+", default=[
        Path("/123090047/nuscene_part04_qwen397b.json"),
        Path("/123090047/nuscene_part05_qwen397b.json"),
        Path("/123090047/nuscene_part06_qwen397b.json"),
    ])
    p.add_argument("--model-dir", type=Path, default=root / "models" / "Cosmos-Reason1-7B")
    p.add_argument("--output-dir", type=Path, default=root / "cosmos_extract_output")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--frames-per-camera", type=int, default=0,
                   help="0 means use all frames in each clip")
    p.add_argument("--max-clips", type=int, default=0,
                   help="0 means all clips")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS_1BASED,
        help="1-based decoder layer indices to extract MLP outputs from",
    )
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs in --output-dir if present",
    )
    return p.parse_args()


def select_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return []
    if k <= 0 or k >= n_frames:
        return list(range(n_frames))
    if k == 1:
        return [n_frames // 2]
    return np.linspace(0, n_frames - 1, num=k, dtype=int).tolist()


def extract_archives(parts: list[int], tgz_dir: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    for part in parts:
        tgz_path = tgz_dir / f"v1.0-trainval{part:02d}_keyframes.tgz"
        if not tgz_path.exists():
            raise FileNotFoundError(f"Missing archive: {tgz_path}")

        marker = extract_root / f".part_{part:02d}.done"
        marker_state = f"{tgz_path.stat().st_size}:{int(tgz_path.stat().st_mtime)}"
        if marker.exists() and marker.read_text().strip() == marker_state:
            print(f"[untar] part {part:02d} already extracted, skip")
            continue

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


def collect_multiview_images_for_sample(
    sample: dict,
    extract_root: Path,
) -> tuple[list[Image.Image], dict[str, int], int | None]:
    """Collect the six camera views for a single flat sample (frame) in CAMERAS order."""
    images_dict = sample.get("images", {})
    images: list[Image.Image] = []
    n_images_per_camera: dict[str, int] = {camera: 0 for camera in CAMERAS}

    for camera in CAMERAS:
        raw = images_dict.get(camera)
        local_path = channel_path_to_local(raw, extract_root) if raw else None
        if local_path is None or not local_path.exists():
            continue
        images.append(Image.open(local_path).convert("RGB"))
        n_images_per_camera[camera] = 1

    return images, n_images_per_camera, sample.get("timestamp")


def summarize_tokens(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3:
        raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    tokens = hidden[0]
    return tokens.mean(dim=0), tokens[-1]


def save_checkpoint(
    output_dir: Path,
    layers: list[int],
    out_mean_by_layer: dict[int, list[np.ndarray]],
    out_last_by_layer: dict[int, list[np.ndarray]],
    texts: list[dict],
    meta: list[dict],
) -> None:
    if not meta:
        return
    for layer_no in layers:
        mean_rows = out_mean_by_layer[layer_no]
        last_rows = out_last_by_layer[layer_no]
        np.save(
            output_dir / f"layer{layer_no:02d}_mlp_output_mean.npy",
            np.stack(mean_rows).astype(np.float32),
        )
        np.save(
            output_dir / f"layer{layer_no:02d}_mlp_output_last_token.npy",
            np.stack(last_rows).astype(np.float32),
        )
    with (output_dir / "final_text_output.jsonl").open("w", encoding="utf-8") as f:
        for row in texts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_checkpoint(
    output_dir: Path,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], list[dict], list[dict]]:
    raise RuntimeError("load_checkpoint must be called with layers via load_checkpoint_for_layers")


def load_checkpoint_for_layers(
    output_dir: Path,
    layers: list[int],
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], list[dict], list[dict]]:
    text_path = output_dir / "final_text_output.jsonl"
    meta_path = output_dir / "meta.json"

    if not (text_path.exists() and meta_path.exists()):
        return ({layer_no: [] for layer_no in layers}, {layer_no: [] for layer_no in layers}, [], [])

    per_layer_mean_np: dict[int, np.ndarray] = {}
    per_layer_last_np: dict[int, np.ndarray] = {}
    for layer_no in layers:
        mean_path = output_dir / f"layer{layer_no:02d}_mlp_output_mean.npy"
        last_path = output_dir / f"layer{layer_no:02d}_mlp_output_last_token.npy"
        if not (mean_path.exists() and last_path.exists()):
            return ({layer_no: [] for layer_no in layers}, {layer_no: [] for layer_no in layers}, [], [])
        per_layer_mean_np[layer_no] = np.load(mean_path)
        per_layer_last_np[layer_no] = np.load(last_path)

    with text_path.open("r", encoding="utf-8") as f:
        text_rows = [json.loads(line) for line in f if line.strip()]
    with meta_path.open("r", encoding="utf-8") as f:
        meta_rows = json.load(f)

    n = len(meta_rows)
    if len(text_rows) != n:
        raise ValueError(f"Checkpoint row-count mismatch: text={len(text_rows)}, meta={n}")

    for layer_no in layers:
        if per_layer_mean_np[layer_no].shape[0] != n or per_layer_last_np[layer_no].shape[0] != n:
            raise ValueError(
                "Checkpoint row-count mismatch for layer "
                f"{layer_no}: mean={per_layer_mean_np[layer_no].shape[0]}, "
                f"last={per_layer_last_np[layer_no].shape[0]}, meta={n}"
            )

    out_mean_by_layer = {
        layer_no: [per_layer_mean_np[layer_no][i] for i in range(n)]
        for layer_no in layers
    }
    out_last_by_layer = {
        layer_no: [per_layer_last_np[layer_no][i] for i in range(n)]
        for layer_no in layers
    }
    return out_mean_by_layer, out_last_by_layer, text_rows, meta_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Step 1: images must be extracted before model inference.
    extract_archives(args.parts, args.tgz_dir, args.extract_root)

    all_samples = []
    for ann_path in args.annotations:
        with Path(ann_path).open("r", encoding="utf-8") as f:
            all_samples.extend(json.load(f).get("samples", []))

    if args.max_clips > 0:
        all_samples = all_samples[: args.max_clips]
    
    print(f"[data] samples to process: {len(all_samples)}")

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
    n_model_layers = len(model.model.language_model.layers)
    selected_layers = sorted(set(args.layers))
    layer_to_model_idx: dict[int, int] = {}
    for layer_no in selected_layers:
        if layer_no < 1:
            raise ValueError(f"Invalid 1-based layer index {layer_no}; must be >= 1")
        if layer_no <= n_model_layers:
            layer_to_model_idx[layer_no] = layer_no - 1
            continue
        # Keep compatibility with workflows that refer to a conceptual "32nd" decoder block.
        # On this Cosmos checkpoint, the text decoder has 28 layers, so map 32 -> last layer.
        if layer_no == 32:
            layer_to_model_idx[layer_no] = n_model_layers - 1
            print(
                f"[warn] requested layer 32 but model has {n_model_layers} decoder layers; "
                f"mapping layer 32 -> model layer {n_model_layers}"
            )
            continue
        raise ValueError(
            f"Invalid 1-based layer index {layer_no}; valid range is [1, {n_model_layers}] "
            f"(32 is specially mapped to last layer)"
        )

    captures: dict[int, torch.Tensor] = {}

    def make_mlp_out_hook(layer_no: int):
        def _hook(_module, _inputs, output):
            captures[layer_no] = output.detach()

        return _hook

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for layer_no in selected_layers:
        layer_idx = layer_to_model_idx[layer_no]
        mlp = model.model.language_model.layers[layer_idx].mlp
        hooks.append(mlp.register_forward_hook(make_mlp_out_hook(layer_no)))

    out_dim = model.config.hidden_size

    mlp_out_mean_rows_by_layer: dict[int, list[np.ndarray]] = {ln: [] for ln in selected_layers}
    mlp_out_last_rows_by_layer: dict[int, list[np.ndarray]] = {ln: [] for ln in selected_layers}
    text_rows: list[dict] = []
    meta_rows: list[dict] = []
    start_clip_idx = 0

    if args.resume:
        (
            mlp_out_mean_rows_by_layer,
            mlp_out_last_rows_by_layer,
            text_rows,
            meta_rows,
        ) = load_checkpoint_for_layers(args.output_dir, selected_layers)
        if meta_rows:
            # Resume from the next clip after the last saved global_clip_index.
            start_clip_idx = len(meta_rows)
            print(f"[resume] found {len(meta_rows)} saved rows, continuing from sample index {start_clip_idx}/{len(all_samples)}")
        else:
            print("[resume] no existing checkpoint rows found, starting from scratch")

    skipped = max(start_clip_idx - len(meta_rows), 0)

    try:
        for idx, sample in enumerate(all_samples[start_clip_idx:], start=start_clip_idx + 1):
            clip_out_mean_by_layer: dict[int, np.ndarray] = {
                    ln: np.zeros((out_dim,), dtype=np.float32)
                for ln in selected_layers
            }
            clip_out_last_by_layer: dict[int, np.ndarray] = {
                    ln: np.zeros((out_dim,), dtype=np.float32)
                for ln in selected_layers
            }

            images, n_images_per_camera, frame_timestamp = collect_multiview_images_for_sample(
                sample, args.extract_root
            )
            if len(images) != len(CAMERAS):
                for im in images:
                    im.close()
                skipped += 1
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
                model(**inputs)
            missing_layers = [ln for ln in selected_layers if ln not in captures]
            if missing_layers:
                for im in images:
                    im.close()
                raise RuntimeError(f"Failed to capture MLP hooks for layers: {missing_layers}")

            for layer_no in selected_layers:
                pooled_out_mean, pooled_out_last = summarize_tokens(captures[layer_no])
                clip_out_mean_by_layer[layer_no] = pooled_out_mean.float().cpu().numpy()
                clip_out_last_by_layer[layer_no] = pooled_out_last.float().cpu().numpy()

            captures.clear()

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], gen_ids, strict=False)
            ]
            gen_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            for im in images:
                im.close()

            for layer_no in selected_layers:
                mlp_out_mean_rows_by_layer[layer_no].append(clip_out_mean_by_layer[layer_no])
                mlp_out_last_rows_by_layer[layer_no].append(clip_out_last_by_layer[layer_no])
            text_rows.append(
                {
                    "scene_name": sample.get("scene_name"),
                    "scene_token": sample.get("scene_token"),
                    "sample_token": sample.get("sample_token"),
                    "timestamp": frame_timestamp,
                    "prompt": args.prompt,
                    "multiview_output": gen_text,
                }
            )
            meta_rows.append(
                {
                    "row_index": len(meta_rows),
                    "scene_name": sample.get("scene_name"),
                    "scene_token": sample.get("scene_token"),
                    "sample_token": sample.get("sample_token"),
                    "timestamp": frame_timestamp,
                    "n_images_per_camera": n_images_per_camera,
                    "valid_camera_count": len(CAMERAS),
                }
            )
            print(f"[progress] Scene {sample.get('scene_name')} | Frame {idx}/{len(all_samples)} processed. Total saved={len(meta_rows)}")

            if idx % args.save_every == 0:
                save_checkpoint(
                    args.output_dir,
                    selected_layers,
                    mlp_out_mean_rows_by_layer,
                    mlp_out_last_rows_by_layer,
                    text_rows,
                    meta_rows,
                )
                print(f"[progress] {idx}/{len(all_samples)} processed, kept={len(meta_rows)}, skipped={skipped}")

    finally:
        for h in hooks:
            h.remove()

    save_checkpoint(
        args.output_dir,
        selected_layers,
        mlp_out_mean_rows_by_layer,
        mlp_out_last_rows_by_layer,
        text_rows,
        meta_rows,
    )
    print("[done] extraction finished")
    print(f"[done] kept={len(meta_rows)} skipped={skipped}")
    for layer_no in selected_layers:
        print(f"[done] file: {args.output_dir / f'layer{layer_no:02d}_mlp_output_mean.npy'}")
        print(f"[done] file: {args.output_dir / f'layer{layer_no:02d}_mlp_output_last_token.npy'}")
    print(f"[done] file: {args.output_dir / 'final_text_output.jsonl'}")


if __name__ == "__main__":
    main()

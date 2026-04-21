#!/usr/bin/env python3
"""
Long-Tail Driving Scene Discovery — End-to-End Pipeline (V2)

Optimizations over V1:
  1. Multi-frame input: configurable frames per camera (default 4, was 1)
  2. Higher resolution: max_pixels=401408 (~633×633, was 316×316)
  3. Feature-level augmentation: mixup + noise + tail oversampling
  4. Deeper SAE encoder: 2-layer MLP with GELU + residual
  5. Semantic interpretation: trace top z_t neurons to scene semantics

Steps:
  1. Build image whitelist from selected tgz parts (default: 01-03)
  2. Extract features: Cosmos Reason1-7B hidden states *before* the last MLP
  3. Pre-SAE activation analysis (raw feature baseline)
  4. Train DeepLongTailSAE with pairwise margin loss + augmentation
  5. Post-SAE activation analysis
  6. Semantic interpretation of top z_t neurons
  7. Pre-SAE vs Post-SAE comparison summary

Usage:
  CUDA_VISIBLE_DEVICES=0 python pipeline.py --parts 1 2 3
  python pipeline.py --skip-extract   # if features already cached
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models" / "Cosmos-Reason1-7B"
CLIPS_JSON = DATASET_DIR / "nuscenes-annotation-platform" / "vlm_annotated_clips.json"
TGZ_DIR = DATASET_DIR / "nuscenes"
IMAGE_DIR = DATASET_DIR / "nuscenes" / "nuscenes_extracted" / "samples"

OLD_PATH_PREFIX = "/data2/visitor/czh/nuscenes_keyframes/"
NEW_PATH_PREFIX = str(DATASET_DIR / "nuscenes") + "/"

OUTPUT_DIR = SCRIPT_DIR / "pipeline_output"

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
SEED = 42
NORMAL_MAX = 2.0
TAIL_MIN = 4.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3],
                   help="Which tgz parts to use (e.g. 1 2 3)")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip feature extraction, use cached features")
    # [OPT-2] Higher resolution: 1003520 ≈ 1002×1002 (was 100352 ≈ 316×316)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--min-pixels", type=int, default=3136)
    # [OPT-1] Multi-frame: 16 frames per camera (was 1)
    p.add_argument("--frames-per-camera", type=int, default=16)
    p.add_argument("--prompt", type=str,
                   default="Analyze this driving scene. Describe the weather, lighting, "
                           "road type, traffic density, and any unusual or rare objects, "
                           "behaviors, or events such as construction, accidents, jaywalking, "
                           "emergency vehicles, animals, or adverse conditions.")
    p.add_argument("--save-every", type=int, default=50)
    # SAE hyperparameters
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--tail-ratio", type=float, default=0.3)
    p.add_argument("--topk-ratio", type=float, default=0.15)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--delta-l2", type=float, default=2.0)
    p.add_argument("--gamma-tail", type=float, default=0.35)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=40)
    # [OPT-3] Augmentation params
    p.add_argument("--aug-mixup-alpha", type=float, default=0.3,
                   help="Mixup alpha for Beta distribution (0=disabled)")
    p.add_argument("--aug-noise-std", type=float, default=0.05,
                   help="Gaussian noise std for tail samples (0=disabled)")
    p.add_argument("--aug-oversample", type=float, default=2.0,
                   help="Oversample tail class by this factor (1=no oversampling)")
    # [OPT-4] Deeper encoder
    p.add_argument("--encoder-layers", type=int, default=2,
                   help="Number of encoder layers (1=original, 2-3=deeper)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Build image whitelist from tgz parts
# ═══════════════════════════════════════════════════════════════════════════════

def build_image_whitelist(parts: list[int]) -> set[str]:
    """Build whitelist from tgz contents, or fall back to scanning extracted dir."""
    whitelist = set()

    # Try tgz listing first
    any_tgz = False
    for part_num in parts:
        tgz_path = TGZ_DIR / f"v1.0-trainval{part_num:02d}_keyframes.tgz"
        if not tgz_path.exists():
            continue
        any_tgz = True
        print(f"  Listing {tgz_path.name} ...")
        result = subprocess.run(
            ["tar", "tzf", str(tgz_path)],
            capture_output=True, text=True, timeout=300,
        )
        for line in result.stdout.splitlines():
            if line.endswith(".jpg") and line.startswith("samples/"):
                rel = line[len("samples/"):]
                whitelist.add(rel)

    # Fallback: scan extracted directory on disk
    if not any_tgz and IMAGE_DIR.exists():
        print(f"  No tgz files found, scanning extracted images at {IMAGE_DIR} ...")
        for cam_dir in IMAGE_DIR.iterdir():
            if not cam_dir.is_dir():
                continue
            for img_file in cam_dir.iterdir():
                if img_file.suffix == ".jpg":
                    whitelist.add(f"{cam_dir.name}/{img_file.name}")

    print(f"  Whitelist: {len(whitelist)} images")
    return whitelist


def remap_path(p: str) -> str:
    if p.startswith(OLD_PATH_PREFIX):
        return NEW_PATH_PREFIX + p[len(OLD_PATH_PREFIX):]
    return p


def get_image_rel_path(full_path: str) -> str | None:
    idx = full_path.find("samples/")
    if idx < 0:
        return None
    return full_path[idx + len("samples/"):]


def filter_clips(clips: list[dict], whitelist: set[str]) -> list[dict]:
    """Keep only clips where ALL camera frames of the middle frame are in the whitelist."""
    filtered = []
    for clip in clips:
        frames = clip.get("frames", [])
        if not frames:
            continue
        mid_frame = frames[len(frames) // 2]
        channels = mid_frame.get("channels", {})
        all_present = True
        for cam in CAMERAS:
            raw_path = channels.get(cam)
            if raw_path is None:
                all_present = False
                break
            remapped = remap_path(raw_path)
            rel = get_image_rel_path(remapped)
            if rel is None or rel not in whitelist:
                all_present = False
                break
        if all_present:
            filtered.append(clip)
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Feature extraction — Multi-frame + Higher resolution [OPT-1, OPT-2]
# ═══════════════════════════════════════════════════════════════════════════════

def load_cosmos_model(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading processor from {MODEL_DIR} ...")
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )

    dt = torch.bfloat16
    print(f"Loading model (bf16) | max_pixels={args.max_pixels} "
          f"(~{int(args.max_pixels**0.5)}×{int(args.max_pixels**0.5)}) ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), torch_dtype=dt, device_map="cuda:0",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.0f}s | "
          f"GPU free: {torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB")
    return model, processor


_pre_mlp_capture = {}


def _hook_fn(module, input, output):
    """Capture the input to the last MLP (= hidden state after attention, before MLP)."""
    _pre_mlp_capture["hidden"] = input[0].detach()


def _select_frame_indices(n_frames: int, frames_per_camera: int) -> list[int]:
    """Select evenly-spaced frame indices from n_frames available."""
    if frames_per_camera >= n_frames:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, frames_per_camera, dtype=int).tolist()


@torch.no_grad()
def extract_camera_feature(model, processor, images: list[Image.Image],
                           prompt: str, image_token_id: int,
                           hook_handle) -> np.ndarray:
    """
    [OPT-1] Multi-frame: accepts a LIST of images (multiple frames from same camera).
    Feeds all images into one prompt, mean-pools visual tokens across all frames.
    [OPT-2] Higher resolution handled by processor's max_pixels setting.
    """
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    _pre_mlp_capture.clear()
    model.model(**inputs)

    pre_mlp_hidden = _pre_mlp_capture["hidden"]  # [1, seq_len, 3584]

    input_ids = inputs["input_ids"][0]
    visual_mask = (input_ids == image_token_id)
    if visual_mask.any():
        visual_features = pre_mlp_hidden[0, visual_mask]
        feature = visual_features.mean(dim=0)
    else:
        feature = pre_mlp_hidden[0].mean(dim=0)

    del pre_mlp_hidden, inputs
    torch.cuda.empty_cache()

    return feature.float().cpu().numpy()


def extract_all_features(clips: list[dict], args) -> list[dict]:
    """Extract features for all clips with multi-frame support."""
    model, processor = load_cosmos_model(args)
    image_token_id = model.config.image_token_id

    last_mlp = model.model.language_model.layers[-1].mlp
    hook_handle = last_mlp.register_forward_hook(_hook_fn)

    results = []
    ckpt_path = OUTPUT_DIR / "extract_checkpoint.json"

    if ckpt_path.exists():
        print(f"Found extraction checkpoint, loading ...")
        with open(ckpt_path) as f:
            results = json.load(f)
        done_indices = {r["global_clip_index"] for r in results}
        clips = [c for c in clips if c["global_clip_index"] not in done_indices]
        print(f"  Resuming: {len(results)} done, {len(clips)} remaining")

    print(f"  Multi-frame: {args.frames_per_camera} frames/camera | "
          f"Resolution: max_pixels={args.max_pixels}")

    t_start = time.time()
    for i, clip in enumerate(clips):
        t0 = time.time()
        frames = clip["frames"]
        selected_indices = _select_frame_indices(len(frames), args.frames_per_camera)

        camera_embeddings = {}
        cam_features = []

        for cam in CAMERAS:
            cam_images = []
            for fi in selected_indices:
                raw_path = frames[fi].get("channels", {}).get(cam)
                if raw_path is None:
                    continue
                img_path = remap_path(raw_path)
                if os.path.exists(img_path):
                    cam_images.append(Image.open(img_path).convert("RGB"))

            if not cam_images:
                camera_embeddings[cam] = [0.0] * model.config.hidden_size
                cam_features.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue

            feat = extract_camera_feature(model, processor, cam_images, args.prompt,
                                          image_token_id, hook_handle)
            camera_embeddings[cam] = feat.tolist()
            cam_features.append(feat)

        clip_embedding = np.mean(cam_features, axis=0).tolist()

        results.append({
            "global_clip_index": clip["global_clip_index"],
            "scene_name": clip.get("scene_name"),
            "scene_token": clip.get("scene_token"),
            "tail_score": clip.get("tail_score"),
            "description": clip.get("description", ""),
            "annotation": clip.get("annotation", ""),
            "camera_embeddings": camera_embeddings,
            "clip_embedding": clip_embedding,
            "n_frames_used": len(selected_indices),
        })

        elapsed = time.time() - t0
        total_remaining = len(clips) - (i + 1)
        avg_time = (time.time() - t_start) / (i + 1)
        eta_min = total_remaining * avg_time / 60

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{len(results)}/{len(results) + total_remaining}] "
                  f"clip {clip['global_clip_index']} | "
                  f"{elapsed:.1f}s | avg {avg_time:.1f}s/clip | ETA {eta_min:.0f}min")

        if (i + 1) % args.save_every == 0:
            ckpt_path.write_text(json.dumps(results, ensure_ascii=False))

    hook_handle.remove()
    del model, processor
    torch.cuda.empty_cache()

    if ckpt_path.exists():
        ckpt_path.unlink()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3: Pre-SAE Activation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def pre_sae_analysis(features: np.ndarray, scores: np.ndarray) -> dict:
    """Analyze raw Cosmos features before SAE. Returns metrics dict."""
    print("\n" + "=" * 70)
    print("  STEP 3: Pre-SAE Activation Analysis (Raw Cosmos Features)")
    print("=" * 70)

    is_normal = scores <= NORMAL_MAX
    is_tail = scores >= TAIL_MIN
    labeled = is_normal | is_tail

    n_normal = is_normal.sum()
    n_tail = is_tail.sum()
    print(f"\nSamples: {len(features)} total | {n_normal} normal | {n_tail} tail")
    print(f"Feature dim: {features.shape[1]}")

    for name, mask in [("Normal", is_normal), ("Tail", is_tail)]:
        if mask.sum() == 0:
            continue
        data = features[mask]
        l2 = np.linalg.norm(data, axis=1)
        nz = (np.abs(data) > 0.01).mean()
        print(f"\n{name} samples (n={mask.sum()}):")
        print(f"  Mean value:     {data.mean():.4f}")
        print(f"  Std:            {data.std():.4f}")
        print(f"  L2 norm:        {l2.mean():.4f} +/- {l2.std():.4f}")
        print(f"  Non-zero ratio: {nz * 100:.2f}%")
        print(f"  Value range:    [{data.min():.4f}, {data.max():.4f}]")

    l2_norms = np.linalg.norm(features, axis=1)
    labels = is_tail[labeled].astype(int)
    auc_l2 = roc_auc_score(labels, l2_norms[labeled])
    print(f"\nRaw L2 norm AUC (naive): {auc_l2:.4f}")

    mean_tail = features[is_tail].mean(axis=0) if is_tail.any() else np.zeros(features.shape[1])
    mean_normal = features[is_normal].mean(axis=0) if is_normal.any() else np.zeros(features.shape[1])
    diff_vec = mean_tail - mean_normal
    diff_score = features @ diff_vec
    auc_diff = roc_auc_score(labels, diff_score[labeled])
    print(f"Mean-diff projection AUC: {auc_diff:.4f}")

    X_train = features[labeled]
    y_train = labels
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_train)[:, 1]
    auc_lr = roc_auc_score(y_train, proba)
    print(f"Logistic Regression AUC (on labeled set): {auc_lr:.4f}")

    importance = np.abs(diff_vec)
    top_dims = np.argsort(-importance)[:10]
    print(f"\nTop 10 discriminative dimensions (by |mean_tail - mean_normal|):")
    for rank, d in enumerate(top_dims):
        print(f"  #{rank + 1} dim[{d}]: tail_mean={mean_tail[d]:.4f}, "
              f"normal_mean={mean_normal[d]:.4f}, diff={diff_vec[d]:.4f}")

    metrics = {
        "pre_sae_auc_l2": float(auc_l2),
        "pre_sae_auc_diff": float(auc_diff),
        "pre_sae_auc_lr": float(auc_lr),
        "pre_sae_l2_normal": float(np.linalg.norm(features[is_normal], axis=1).mean()),
        "pre_sae_l2_tail": float(np.linalg.norm(features[is_tail], axis=1).mean()),
    }
    print(f"\nPre-SAE summary: L2_AUC={auc_l2:.4f}, Diff_AUC={auc_diff:.4f}, LR_AUC={auc_lr:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4: SAE Training — Deeper encoder + GELU + Augmentation [OPT-3, OPT-4]
# ═══════════════════════════════════════════════════════════════════════════════

class DeepLongTailSAE(nn.Module):
    """
    [OPT-4] Deeper SAE encoder with GELU activation and optional residual.
    Supports 1-3 encoder layers. Decoder remains single-layer (standard for SAE).
    """
    def __init__(self, input_dim, hidden_dim, tail_ratio=0.5,
                 global_topk_ratio=0.15, dropout=0.2, encoder_layers=2):
        super().__init__()
        self.tail_dim = int(hidden_dim * tail_ratio)
        self.normal_dim = hidden_dim - self.tail_dim
        self.global_topk_k = max(1, int(hidden_dim * global_topk_ratio))

        layers = []
        in_d = input_dim
        for i in range(encoder_layers):
            out_d = hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.GELU())
            if i < encoder_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.encoder = nn.Sequential(*layers)

        self.use_residual = (encoder_layers >= 2 and input_dim == hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def global_topk(self, z):
        if self.global_topk_k >= z.size(-1):
            return z
        vals, idx = torch.topk(z, self.global_topk_k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, idx, vals)
        return sparse

    def encode(self, x):
        z = self.encoder(x)
        if self.use_residual:
            z = z + x
        z = z * self.scale
        z = self.global_topk(z)
        z = self.dropout_layer(z)
        z_n = z[:, :self.normal_dim]
        z_t = z[:, self.normal_dim:]
        return z, z_n, z_t

    def forward(self, x):
        z, z_n, z_t = self.encode(x)
        return self.decoder(z), z, z_n, z_t


def augment_tail_features(features: np.ndarray, scores: np.ndarray, args) -> tuple:
    """
    [OPT-3] Feature-level augmentation to handle class imbalance.
    1. Tail oversampling: duplicate tail features
    2. Mixup: interpolate between pairs of tail samples
    3. Gaussian noise: add noise to tail samples
    """
    is_tail = scores >= TAIL_MIN
    tail_feats = features[is_tail]
    tail_scores = scores[is_tail]
    n_tail = len(tail_feats)

    if n_tail == 0:
        return features, scores

    aug_feats = []
    aug_scores = []

    # Tail oversampling
    n_oversample = int(n_tail * max(0, args.aug_oversample - 1))
    if n_oversample > 0:
        oversample_idx = np.random.choice(n_tail, n_oversample, replace=True)
        aug_feats.append(tail_feats[oversample_idx])
        aug_scores.append(tail_scores[oversample_idx])

    # Mixup between tail samples
    if args.aug_mixup_alpha > 0 and n_tail >= 2:
        n_mixup = n_tail
        idx_a = np.random.choice(n_tail, n_mixup)
        idx_b = np.random.choice(n_tail, n_mixup)
        lam = np.random.beta(args.aug_mixup_alpha, args.aug_mixup_alpha, size=(n_mixup, 1))
        mixed = lam * tail_feats[idx_a] + (1 - lam) * tail_feats[idx_b]
        mixed_scores = np.maximum(tail_scores[idx_a], tail_scores[idx_b])
        aug_feats.append(mixed.astype(np.float32))
        aug_scores.append(mixed_scores)

    # Gaussian noise on tail samples
    if args.aug_noise_std > 0:
        noise = np.random.randn(*tail_feats.shape).astype(np.float32) * args.aug_noise_std
        noisy = tail_feats + noise
        aug_feats.append(noisy)
        aug_scores.append(tail_scores)

    if aug_feats:
        all_feats = np.concatenate([features] + aug_feats, axis=0)
        all_scores = np.concatenate([scores] + aug_scores, axis=0)
        n_aug = len(all_feats) - len(features)
        n_new_tail = sum(1 for s in all_scores[len(features):] if s >= TAIL_MIN)
        print(f"  Augmentation: +{n_aug} samples ({n_new_tail} tail) | "
              f"Total: {len(all_feats)} (was {len(features)})")
        return all_feats, all_scores

    return features, scores


def train_sae(features: np.ndarray, scores: np.ndarray, args,
              clip_info: list[dict] | None = None) -> tuple:
    """Train SAE with augmentation and deeper architecture."""
    print("\n" + "=" * 70)
    print("  STEP 4: SAE Training (DeepLongTailSAE + Augmentation)")
    print("=" * 70)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device(args.device)
    N, D = features.shape

    # Scene-level split (on ORIGINAL data, before augmentation)
    idx = list(range(N))
    random.shuffle(idx)
    split = int(N * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    n_train_normal = sum(1 for i in train_idx if scores[i] <= NORMAL_MAX)
    n_train_tail = sum(1 for i in train_idx if scores[i] >= TAIL_MIN)
    n_val_normal = sum(1 for i in val_idx if scores[i] <= NORMAL_MAX)
    n_val_tail = sum(1 for i in val_idx if scores[i] >= TAIL_MIN)
    print(f"Split: train={len(train_idx)} (normal={n_train_normal}, tail={n_train_tail}) | "
          f"val={len(val_idx)} (normal={n_val_normal}, tail={n_val_tail})")

    # [OPT-3] Augment training features
    train_feats_orig = features[train_idx]
    train_scores_orig = scores[train_idx]
    train_feats_aug, train_scores_aug = augment_tail_features(
        train_feats_orig, train_scores_orig, args)

    # Z-score normalization (fit on original train data)
    feat_mean = torch.tensor(train_feats_orig.mean(axis=0), dtype=torch.float32)
    feat_std = torch.tensor(train_feats_orig.std(axis=0), dtype=torch.float32).clamp(min=1e-6)

    train_norm = torch.tensor(
        (train_feats_aug - feat_mean.numpy()) / feat_std.numpy(),
        dtype=torch.float32).to(device)
    train_score_t = torch.tensor(train_scores_aug, dtype=torch.float32).to(device)

    val_feats = features[val_idx]
    val_norm = torch.tensor(
        (val_feats - feat_mean.numpy()) / feat_std.numpy(),
        dtype=torch.float32).to(device)
    val_score_t = torch.tensor(scores[val_idx], dtype=torch.float32).to(device)

    # Also prepare full normalized data for post-analysis
    all_norm = torch.tensor(
        (features - feat_mean.numpy()) / feat_std.numpy(),
        dtype=torch.float32).to(device)
    all_score_t = torch.tensor(scores, dtype=torch.float32).to(device)

    # [OPT-4] Deeper SAE
    sae = DeepLongTailSAE(
        D, args.hidden_dim, args.tail_ratio, args.topk_ratio,
        encoder_layers=args.encoder_layers,
    ).to(device)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"SAE: {D} -> {args.hidden_dim} (normal={sae.normal_dim}, tail={sae.tail_dim}) | "
          f"topk={sae.global_topk_k} | encoder_layers={args.encoder_layers} | "
          f"GELU | {n_params:,} params")

    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_epoch, wait = 0.0, 0, 0
    best_state = None
    n_train = len(train_norm)

    for epoch in range(1, args.epochs + 1):
        sae.train()
        perm = torch.randperm(n_train, device=device)
        total_loss, n_batches = 0.0, 0

        for s in range(0, n_train, args.batch_size):
            bi = perm[s:s + args.batch_size]
            bx = train_norm[bi]
            bs = train_score_t[bi]

            x_hat, z, z_n, z_t = sae(bx)
            is_n = bs <= NORMAL_MAX
            is_t = bs >= TAIL_MIN
            z_t_l2 = z_t.norm(dim=1, p=2)

            loss = F.mse_loss(x_hat, bx)
            loss = loss + 0.20 * z_n.abs().mean() + 0.25 * z_n.pow(2).mean()
            if is_t.any():
                loss = loss + args.gamma_tail * F.relu(args.delta_l2 - z_t_l2[is_t]).pow(2).mean()
            if is_n.any():
                loss = loss + args.gamma_tail * z_t_l2[is_n].pow(2).mean()
            if is_t.any() and is_n.any():
                loss = loss + 0.5 * F.relu(args.margin - (z_t_l2[is_t].mean() - z_t_l2[is_n].mean()))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        sched.step()

        # Validation AUC
        sae.eval()
        with torch.no_grad():
            _, _, _, vzt = sae(val_norm)
            vl2 = vzt.norm(dim=1, p=2).cpu().numpy()
            vs = val_score_t.cpu().numpy()
            is_n_v = vs <= NORMAL_MAX
            is_t_v = vs >= TAIL_MIN
            lab = is_n_v | is_t_v
            if lab.sum() >= 2 and is_t_v[lab].sum() > 0 and is_n_v[lab].sum() > 0:
                auc = roc_auc_score(is_t_v[lab], vl2[lab])
            else:
                auc = 0.5

        is_best = auc > best_auc + 1e-4
        if is_best:
            best_auc = auc
            best_epoch = epoch
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
        else:
            wait += 1

        if epoch == 1 or epoch % 20 == 0 or is_best:
            mark = " *BEST*" if is_best else ""
            print(f"  E{epoch:03d} loss={total_loss / max(n_batches, 1):.4f} | "
                  f"AUC={auc:.4f} | best={best_auc:.4f}@{best_epoch} wait={wait}{mark}")

        if wait >= args.patience and epoch >= 60:
            print(f"  Early stopping at epoch {epoch}")
            break

    sae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    sae.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sae_state": sae.state_dict(),
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "feat_mean": feat_mean.cpu(),
        "feat_std": feat_std.cpu(),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }, OUTPUT_DIR / "best_model.pth")

    return sae, all_norm, all_score_t, train_idx, val_idx, feat_mean, feat_std


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5: Post-SAE Activation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def print_activation_stats(label: str, data: np.ndarray) -> np.ndarray:
    l2 = np.linalg.norm(data, axis=1)
    nz_ratio = (data > 0.01).mean()
    print(f"\n  {label} (n={len(data)}):")
    print(f"    Mean activation (global): {data.mean():.4f}")
    print(f"    Max activation (global):  {data.max():.4f}")
    print(f"    Per-sample mean:          {data.mean(axis=1).mean():.4f}")
    print(f"    Non-zero ratio:           {nz_ratio * 100:.2f}%")
    print(f"    L2 norm:                  {l2.mean():.4f} +/- {l2.std():.4f}")
    print(f"    L2 range:                 [{l2.min():.4f}, {np.percentile(l2, 25):.4f}, "
          f"{np.median(l2):.4f}, {np.percentile(l2, 75):.4f}, {l2.max():.4f}]")
    return l2


def post_sae_analysis(sae: DeepLongTailSAE, cam_norm: torch.Tensor,
                      score_tensor: torch.Tensor, val_idx: list[int]) -> dict:
    """Full post-SAE activation analysis."""
    print("\n" + "=" * 70)
    print("  STEP 5: Post-SAE Activation Analysis")
    print("=" * 70)

    sae.eval()
    with torch.no_grad():
        _, _, all_z_n, all_z_t = sae(cam_norm)
        all_z_n = all_z_n.cpu().numpy()
        all_z_t = all_z_t.cpu().numpy()

    scores = score_tensor.cpu().numpy()
    is_normal = scores <= NORMAL_MAX
    is_tail = scores >= TAIL_MIN
    labeled = is_normal | is_tail

    z_t_l2 = np.linalg.norm(all_z_t, axis=1)
    labels = is_tail[labeled].astype(int)

    auc_all = roc_auc_score(labels, z_t_l2[labeled])
    print(f"\n*** Full set AUC (||z_t||_2): {auc_all:.4f} ***")

    val_mask = np.zeros(len(scores), dtype=bool)
    for i in val_idx:
        val_mask[i] = True
    val_labeled = val_mask & labeled
    if val_labeled.sum() > 0:
        auc_val = roc_auc_score(is_tail[val_labeled], z_t_l2[val_labeled])
        print(f"*** Val AUC (||z_t||_2): {auc_val:.4f} ***")
    else:
        auc_val = 0.5

    labels_val = is_tail[val_labeled].astype(int)
    scores_val = z_t_l2[val_labeled]
    prec, rec, thr = precision_recall_curve(labels_val, scores_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    bi = np.argmax(f1s)
    bt = thr[min(bi, len(thr) - 1)]
    preds = (scores_val >= bt).astype(int)
    acc = accuracy_score(labels_val, preds)
    f1 = f1_score(labels_val, preds)
    prec_val = precision_score(labels_val, preds)
    rec_val = recall_score(labels_val, preds)

    print(f"\nVal binary classification (threshold={bt:.4f}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec_val:.4f}")
    print(f"  Recall:    {rec_val:.4f}")

    print(f"\n{'_' * 50}")
    print("[z_n (normal features) statistics]")
    for name, mask in [("Normal samples", is_normal), ("Tail samples", is_tail)]:
        if mask.sum() > 0:
            print_activation_stats(name, all_z_n[mask])

    print(f"\n{'_' * 50}")
    print("[z_t (tail features) statistics]")
    zt_l2_normal = None
    zt_l2_tail = None
    for name, mask in [("Normal samples", is_normal), ("Tail samples", is_tail)]:
        if mask.sum() > 0:
            l2 = print_activation_stats(name, all_z_t[mask])
            if "Normal" in name:
                zt_l2_normal = l2
            else:
                zt_l2_tail = l2

    if zt_l2_normal is not None and zt_l2_tail is not None:
        gap = zt_l2_tail.mean() - zt_l2_normal.mean()
        ratio = zt_l2_tail.mean() / max(zt_l2_normal.mean(), 1e-8)
        print(f"\n  *** z_t separation ***")
        print(f"    Gap (tail - normal):  {gap:.4f}")
        print(f"    Ratio (tail/normal):  {ratio:.2f}x")

    for threshold in [0.01, 0.1]:
        zt_act_tail = (all_z_t[is_tail] > threshold).mean(axis=0)
        zt_act_norm = (all_z_t[is_normal] > threshold).mean(axis=0)
        diff = zt_act_tail - zt_act_norm
        top_idx = np.argsort(-diff)[:10]
        print(f"\n  Top 10 z_t neurons (threshold={threshold}):")
        for rank, idx in enumerate(top_idx):
            print(f"    #{rank + 1} neuron[{idx}]: tail={zt_act_tail[idx] * 100:.1f}% "
                  f"normal={zt_act_norm[idx] * 100:.1f}% diff={diff[idx] * 100:+.1f}%")

    print(f"\n{'_' * 50}")
    print("[z_t L2 norm by tail_score]")
    for s in sorted(set(scores)):
        mask = scores == s
        if mask.sum() == 0:
            continue
        l2s = z_t_l2[mask]
        print(f"  score={s:.1f}: n={mask.sum():4d} | "
              f"z_t L2 = {l2s.mean():.4f} +/- {l2s.std():.4f} | median={np.median(l2s):.4f}")

    return {
        "post_sae_auc_all": float(auc_all),
        "post_sae_auc_val": float(auc_val),
        "post_sae_acc": float(acc),
        "post_sae_f1": float(f1),
        "post_sae_precision": float(prec_val),
        "post_sae_recall": float(rec_val),
        "post_sae_zt_gap": float(zt_l2_tail.mean() - zt_l2_normal.mean()) if zt_l2_tail is not None else 0,
        "post_sae_zt_ratio": float(zt_l2_tail.mean() / max(zt_l2_normal.mean(), 1e-8)) if zt_l2_tail is not None else 0,
        "post_sae_zt_normal_mean": float(zt_l2_normal.mean()) if zt_l2_normal is not None else 0,
        "post_sae_zt_tail_mean": float(zt_l2_tail.mean()) if zt_l2_tail is not None else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6: Semantic Interpretation of z_t Neurons [OPT-5]
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_interpretation(sae: DeepLongTailSAE, cam_norm: torch.Tensor,
                            score_tensor: torch.Tensor,
                            clip_info: list[dict], top_k_neurons: int = 10,
                            top_k_clips: int = 5) -> dict:
    """
    [OPT-5] Trace top z_t neurons back to driving scenes for semantic interpretation.
    For each top neuron, find clips that maximally activate it and report their descriptions.
    """
    print("\n" + "=" * 70)
    print("  STEP 6: Semantic Interpretation of Top z_t Neurons")
    print("=" * 70)

    sae.eval()
    with torch.no_grad():
        _, _, _, all_z_t = sae(cam_norm)
        all_z_t = all_z_t.cpu().numpy()

    scores = score_tensor.cpu().numpy()
    is_normal = scores <= NORMAL_MAX
    is_tail = scores >= TAIL_MIN

    zt_act_tail = (all_z_t[is_tail] > 0.01).mean(axis=0)
    zt_act_norm = (all_z_t[is_normal] > 0.01).mean(axis=0)
    neuron_diff = zt_act_tail - zt_act_norm
    top_neurons = np.argsort(-neuron_diff)[:top_k_neurons]

    interpretation = {}

    for rank, neuron_idx in enumerate(top_neurons):
        activations = all_z_t[:, neuron_idx]
        top_clip_indices = np.argsort(-activations)[:top_k_clips]

        print(f"\n  === Neuron #{rank + 1}: z_t[{neuron_idx}] ===")
        print(f"      Tail activation rate: {zt_act_tail[neuron_idx] * 100:.1f}%")
        print(f"      Normal activation rate: {zt_act_norm[neuron_idx] * 100:.1f}%")
        print(f"      Selectivity: {neuron_diff[neuron_idx] * 100:+.1f}%")

        neuron_clips = []
        for ci in top_clip_indices:
            info = clip_info[ci] if ci < len(clip_info) else {}
            act_val = float(activations[ci])
            ts = float(scores[ci])
            desc = info.get("description", "N/A")
            anno = info.get("annotation", "")
            scene = info.get("scene_name", "N/A")

            label = "TAIL" if ts >= TAIL_MIN else ("NORMAL" if ts <= NORMAL_MAX else "GRAY")
            print(f"      Top clip: score={ts:.0f} ({label}) | activation={act_val:.3f} | "
                  f"scene={scene}")
            if desc and desc != "N/A":
                print(f"        description: {desc[:120]}")
            if anno:
                anno_str = anno if isinstance(anno, str) else str(anno)[:120]
                print(f"        annotation:  {anno_str[:120]}")

            neuron_clips.append({
                "clip_index": int(ci),
                "global_clip_index": info.get("global_clip_index"),
                "tail_score": ts,
                "activation": act_val,
                "scene_name": scene,
                "description": desc[:200] if desc else "",
            })

        # Summarize semantic pattern
        tail_count = sum(1 for c in neuron_clips if c["tail_score"] >= TAIL_MIN)
        mean_score = np.mean([c["tail_score"] for c in neuron_clips])

        descriptions = [c["description"] for c in neuron_clips if c["description"]]
        print(f"      Summary: {tail_count}/{len(neuron_clips)} top clips are tail | "
              f"mean score={mean_score:.1f}")

        interpretation[f"neuron_{neuron_idx}"] = {
            "rank": rank + 1,
            "tail_activation_rate": float(zt_act_tail[neuron_idx]),
            "normal_activation_rate": float(zt_act_norm[neuron_idx]),
            "selectivity": float(neuron_diff[neuron_idx]),
            "top_clips": neuron_clips,
            "tail_ratio_in_top": tail_count / max(len(neuron_clips), 1),
        }

    print(f"\n  {'_' * 50}")
    print(f"  Interpretation complete: analyzed {top_k_neurons} neurons x {top_k_clips} clips")

    return interpretation


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 7: Comparison Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(pre_metrics: dict, post_metrics: dict, args):
    print("\n" + "=" * 70)
    print("  STEP 7: Pre-SAE vs Post-SAE Comparison")
    print("=" * 70)

    print(f"\n  V2 Optimizations applied:")
    print(f"    [OPT-1] Multi-frame:    {args.frames_per_camera} frames/camera (was 1)")
    print(f"    [OPT-2] Resolution:     max_pixels={args.max_pixels} "
          f"(~{int(args.max_pixels**0.5)}x{int(args.max_pixels**0.5)}, was 316x316)")
    print(f"    [OPT-3] Augmentation:   mixup_alpha={args.aug_mixup_alpha}, "
          f"noise_std={args.aug_noise_std}, oversample={args.aug_oversample}x")
    print(f"    [OPT-4] Deeper SAE:     {args.encoder_layers}-layer encoder, GELU activation")
    print(f"    [OPT-5] Interpretation: semantic neuron analysis (see Step 6)")

    print(f"\n{'Metric':<35} {'Pre-SAE':>10} {'Post-SAE':>10} {'Delta':>10}")
    print(f"{'_' * 65}")

    comparisons = [
        ("AUC (raw L2 / z_t L2)", "pre_sae_auc_l2", "post_sae_auc_val"),
        ("AUC (mean-diff / z_t L2)", "pre_sae_auc_diff", "post_sae_auc_val"),
        ("AUC (logistic reg / z_t L2)", "pre_sae_auc_lr", "post_sae_auc_val"),
        ("L2 normal mean", "pre_sae_l2_normal", "post_sae_zt_normal_mean"),
        ("L2 tail mean", "pre_sae_l2_tail", "post_sae_zt_tail_mean"),
    ]

    for label, pre_key, post_key in comparisons:
        pre_val = pre_metrics.get(pre_key, 0)
        post_val = post_metrics.get(post_key, 0)
        delta = post_val - pre_val
        sign = "+" if delta > 0 else ""
        print(f"  {label:<33} {pre_val:>10.4f} {post_val:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n  Post-SAE binary classification:")
    print(f"    Accuracy:  {post_metrics.get('post_sae_acc', 0):.4f}")
    print(f"    F1:        {post_metrics.get('post_sae_f1', 0):.4f}")
    print(f"    Precision: {post_metrics.get('post_sae_precision', 0):.4f}")
    print(f"    Recall:    {post_metrics.get('post_sae_recall', 0):.4f}")
    print(f"\n  z_t separation: gap={post_metrics.get('post_sae_zt_gap', 0):.4f}, "
          f"ratio={post_metrics.get('post_sae_zt_ratio', 0):.2f}x")

    improvement = post_metrics.get("post_sae_auc_val", 0) - pre_metrics.get("pre_sae_auc_l2", 0)
    if improvement > 0.05:
        print(f"\n  >> SAE shows significant improvement (+{improvement:.4f} AUC over raw L2)")
    elif improvement > 0:
        print(f"\n  ~  SAE shows marginal improvement (+{improvement:.4f} AUC)")
    else:
        print(f"\n  !! SAE does not improve over raw features ({improvement:.4f} AUC)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / "features.json"

    if not args.skip_extract:
        # Step 1: Build whitelist
        print("=" * 70)
        print("  STEP 1: Building image whitelist from tgz parts")
        print("=" * 70)
        whitelist = build_image_whitelist(args.parts)

        print(f"\nLoading clips from {CLIPS_JSON} ...")
        with open(CLIPS_JSON) as f:
            all_clips = json.load(f)
        print(f"  Total clips: {len(all_clips)}")

        clips = filter_clips(all_clips, whitelist)
        print(f"  Clips with all images in tgz parts {args.parts}: {len(clips)}")

        clip_scores = [c.get("tail_score", -1) for c in clips]
        n_normal = sum(1 for s in clip_scores if s is not None and s <= NORMAL_MAX)
        n_tail = sum(1 for s in clip_scores if s is not None and s >= TAIL_MIN)
        print(f"  Normal (score<={NORMAL_MAX}): {n_normal} | Tail (score>={TAIL_MIN}): {n_tail}")

        # Step 2: Extract features (multi-frame + high-res)
        print("\n" + "=" * 70)
        print("  STEP 2: Feature Extraction (Multi-frame + High-res)")
        print("=" * 70)
        results = extract_all_features(clips, args)

        features_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nSaved {len(results)} clip features to {features_path}")
    else:
        print("Skipping extraction, loading cached features ...")
        with open(features_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} clips")

    features = np.array([r["clip_embedding"] for r in results], dtype=np.float32)
    scores = np.array([float(r.get("tail_score", -1) or -1) for r in results], dtype=np.float32)
    print(f"\nFeature matrix: {features.shape}")

    # Step 3: Pre-SAE analysis
    pre_metrics = pre_sae_analysis(features, scores)

    # Step 4: SAE training (deeper + augmented)
    sae, cam_norm, score_tensor, train_idx, val_idx, feat_mean, feat_std = \
        train_sae(features, scores, args, clip_info=results)

    # Step 5: Post-SAE analysis
    post_metrics = post_sae_analysis(sae, cam_norm, score_tensor, val_idx)

    # Step 6: Semantic interpretation [OPT-5]
    interpretation = semantic_interpretation(sae, cam_norm, score_tensor, results)

    # Step 7: Comparison
    print_comparison(pre_metrics, post_metrics, args)

    # Save all metrics + interpretation
    all_metrics = {
        **pre_metrics,
        **post_metrics,
        "optimizations": {
            "frames_per_camera": args.frames_per_camera,
            "max_pixels": args.max_pixels,
            "encoder_layers": args.encoder_layers,
            "aug_mixup_alpha": args.aug_mixup_alpha,
            "aug_noise_std": args.aug_noise_std,
            "aug_oversample": args.aug_oversample,
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False))
    (OUTPUT_DIR / "interpretation.json").write_text(
        json.dumps(interpretation, indent=2, ensure_ascii=False))

    print(f"\nAll metrics saved to {OUTPUT_DIR / 'metrics.json'}")
    print(f"Interpretation saved to {OUTPUT_DIR / 'interpretation.json'}")
    print("\n" + "=" * 70)
    print("  Pipeline V2 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

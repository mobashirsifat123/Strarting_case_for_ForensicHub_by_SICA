#!/usr/bin/env python3
"""
dump_sica_predictions.py - FINAL eval prediction dumper for the SICA baseline.

Runs the trained CLIP_LORA_PURE (ViT-L/14, LoRA r=8/alpha=16) over an OpenMMSec
manifest and writes per-image predictions as CSV, so that the aggregation script
can compute ACC/AUC/AP/F1 per split, per domain, per primary faking type, and a
macro-domain average (paper Tables 2/6/7/8 style).

DESIGN (faithful + eval-correct):
  * Model built via the SAME ForensicHub registry path as train.py/test.py, with
    the SAME init_config (ViT-L/14, lora_r=8, lora_alpha=16, lora_qkv=qkv,
    lora_apply_out_proj=True, tune_mode=lora). Checkpoint loaded as ckpt['model']
    (same as test.py:145).
  * Transform replicates AIGCLabelDataset._make_output EXACTLY:
    Image.open(path).convert("RGB").resize((224,224)) -> np.array -> AIGCTransform
    post_transform (CLIP mean/std + ToTensorV2). NOTE: this is PIL resize, NOT
    albumentations Resize - matching label_dataset.py:228. (The earlier proposed
    version used albu.Resize which uses cv2 INTER_LINEAR and would NOT match.)
  * Corrupted/missing images are SKIPPED and recorded (not substituted). This is
    the mission's preferred eval policy: do not change labels, do not substitute,
    record every failed item, report the evaluated denominator. (The dataset's
    __getitem__ substitutes on rare residual failure, which is fine for training
    but would corrupt an eval denominator, so we do not use __getitem__ here.)
  * world_size=1: no torchrun, no dist.init_process_group, no NCCL. Safe alongside
    other users' GPU jobs. Uses one GPU (forward only).

USAGE (PYTHONPATH=ForensicHub, same as train/test):
  PYTHONPATH=/path/to/ForensicHub \
  python dump_sica_predictions.py \
        --checkpoint /path/to/logs/sica_train_1gpu/checkpoint-9.pth \
        --manifest  /path/to/OpenMMSecV2/jsons_v4/total_ood_test.json \
        --out       /path/to/predictions/total_ood_test.csv \
        --path-prefix-from /path/to/manifest/prefix/ \
        --path-prefix-to   /path/to/local/dataset/ \
        --gpu 0

  --path-prefix-from is the leading prefix stored in the manifest's image paths;
  --path-prefix-to is the root where your local copy of the dataset lives.
  Omit both (or pass both empty) if the manifest paths already resolve locally.

OUTPUT CSV columns:
  path, label, probability, logit, pred_class, domain, mani_type, sub_mani_type, ori_dataset
A companion <out>.skipped.json records every failed item with its reason.
Exits nonzero on a fatal error; exits 0 with a summary line.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ForensicHub.registry import MODELS, TRANSFORMS, build_from_registry


def parse_args():
    p = argparse.ArgumentParser(description="Dump per-image SICA predictions to CSV (faithful, eval-correct).")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint-<n>.pth (file).")
    p.add_argument("--manifest", required=True, help="OpenMMSec manifest .json (list of dicts).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--model-name", default="ViT-L/14")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--path-prefix-from", default="",
                   help="Leading prefix stored in the manifest's image paths. "
                        "Empty = no remapping. Must be set together with --path-prefix-to.")
    p.add_argument("--path-prefix-to", default="",
                   help="Local dataset root prefix to substitute for --path-prefix-from. "
                        "Empty = no remapping. Must be set together with --path-prefix-from.")
    p.add_argument("--limit", type=int, default=0, help="Debug: only first N entries (0=all).")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for pred_class (default 0.5, matches IMDLBenCo ImageAccuracy).")
    return p.parse_args()


def main():
    args = parse_args()
    if bool(args.path_prefix_from) != bool(args.path_prefix_to):
        sys.exit("[dump] ERROR: --path-prefix-from and --path-prefix-to must be set "
                 "together (both set, or both empty).")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    t0 = time.time()

    # ---- model (registry path, same as train.py/test.py) ---------------------
    model_args = {"name": "CLIP_LORA_PURE", "init_config": {
        "name": args.model_name, "num_classes": 1, "lora_r": 8, "lora_alpha": 16.0,
        "lora_qkv": "qkv", "lora_apply_out_proj": True, "tune_mode": "lora"}}
    print(f"[dump] building model {model_args['name']} ({args.model_name}) ...", flush=True)
    model = build_from_registry(MODELS, model_args)

    print(f"[dump] loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[dump] WARNING missing keys ({len(missing)}): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[dump] WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    model.to(device).eval()

    # ---- transform: AIGCTransform post_transform (CLIP norm + ToTensor) -------
    # Replicates AIGCLabelDataset._make_output (label_dataset.py:227-233) EXACTLY:
    #   image.resize((image_size,image_size)) -> np.array -> post_transform
    # (common_transform/test_transform is an empty Compose for AIGCTransform.)
    transform = build_from_registry(TRANSFORMS, {"name": "AIGCTransform",
        "init_config": {"output_size": [args.image_size, args.image_size], "norm_type": "clip"}})
    post_transform = transform.get_post_transform()
    isize = args.image_size

    def load_tensor(path):
        # Exact match to _make_output preprocessing (PIL resize, not albu.Resize).
        with Image.open(path) as im:
            im = im.convert("RGB").resize((isize, isize))
            arr = np.array(im)
        return post_transform(image=arr)["image"]

    # ---- read manifest --------------------------------------------------------
    with open(args.manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if args.limit > 0:
        entries = entries[:args.limit]
    n_raw = len(entries)
    print(f"[dump] {n_raw} entries in manifest; device={device}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    skipped = []

    def rewrite(p):
        if args.path_prefix_from and p.startswith(args.path_prefix_from):
            return args.path_prefix_to + p[len(args.path_prefix_from):]
        return p

    # ---- iterate in batches; load in worker threads to overlap NFS I/O --------
    from concurrent.futures import ThreadPoolExecutor

    written = 0
    n_real = n_fake = 0
    max_mem = 0
    i = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["path", "label", "probability", "logit", "pred_class",
                    "domain", "mani_type", "sub_mani_type", "ori_dataset"])
        pool = ThreadPoolExecutor(max_workers=max(1, args.num_workers))
        while i < n_raw:
            batch_entries = entries[i:i + args.batch_size]
            # parallel image load (threads -> no fork/CUDA issue)
            def load_one(e):
                p = rewrite(e["path"])
                try:
                    return p, load_tensor(p), e
                except Exception as ex:
                    return p, None, (e, f"{type(ex).__name__}: {ex}")
            results = list(pool.map(load_one, batch_entries))
            tensors, metas, good_paths = [], [], []
            for r in results:
                p = r[0]
                if r[1] is None:
                    e, reason = r[2]
                    skipped.append({"path": e.get("path"), "reason": reason,
                                    "label": e.get("label"), "domain": e.get("domain")})
                    continue
                tensors.append(r[1]); metas.append(r[2]); good_paths.append(p)
            if not tensors:
                i += args.batch_size
                continue
            batch = torch.stack(tensors).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                out = model(image=batch)
            probs = out["pred_label"].float().reshape(-1).cpu().numpy()
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            if mem > max_mem: max_mem = mem
            for p, e, prob in zip(good_paths, metas, probs):
                prob_f = float(prob)
                prob_c = min(max(prob_f, 1e-12), 1 - 1e-12)
                logit = float(np.log(prob_c / (1 - prob_c)))
                pred_class = 1 if prob_f >= args.threshold else 0
                lbl = int(e.get("label", -1))
                if lbl == 0: n_real += 1
                elif lbl == 1: n_fake += 1
                w.writerow([p, lbl, f"{prob_f:.8f}", f"{logit:.8f}", pred_class,
                            e.get("domain", ""), e.get("mani_type", "") or "",
                            e.get("sub_mani_type", "") or "", e.get("ori_dataset", "") or ""])
                written += 1
            i += args.batch_size
            if (i // args.batch_size) % 25 == 0:
                fcsv.flush()
                print(f"[dump] {i}/{n_raw} processed, {written} written, {len(skipped)} skipped, "
                      f"mem={mem:.0f}MiB", flush=True)
        pool.shutdown(wait=True)

    # ---- companion skipped log ------------------------------------------------
    skip_path = args.out + ".skipped.json"
    with open(skip_path, "w", encoding="utf-8") as f:
        json.dump({"manifest": args.manifest, "n_raw": n_raw, "n_written": written,
                   "n_skipped": len(skipped), "skipped": skipped}, f, indent=2)

    dt = time.time() - t0
    summary = (f"[dump] DONE {os.path.basename(args.out)}: raw={n_raw} written={written} "
               f"skipped={len(skipped)} real={n_real} fake={n_fake} "
               f"max_mem={max_mem:.0f}MiB time={dt:.1f}s -> {args.out}")
    print(summary, flush=True)
    # also write a .summary.json for easy parsing
    with open(args.out + ".summary.json", "w", encoding="utf-8") as f:
        json.dump({"manifest": args.manifest, "checkpoint": args.checkpoint,
                   "n_raw": n_raw, "n_written": written, "n_skipped": len(skipped),
                   "n_real": n_real, "n_fake": n_fake, "max_mem_mib": round(max_mem, 1),
                   "elapsed_s": round(dt, 1), "gpu": args.gpu, "threshold": args.threshold}, f, indent=2)
    if written == 0:
        print("[dump] ERROR: wrote 0 rows - failing nonzero", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()

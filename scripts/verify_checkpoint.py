#!/usr/bin/env python3
"""verify_checkpoint.py - forensic verification of the SICA baseline checkpoint.

CPU-ONLY (forces CUDA_VISIBLE_DEVICES=""). Does NOT touch any GPU, so it is safe
to run while the evaluation is in progress.

Produces:
  * a JSON record  (snapshots/checkpoint_verification.json)
  * a human-readable markdown record (snapshots/checkpoint_verification.md)

Records: absolute path, size, full SHA-256, epoch, top-level keys, model tensor
count, presence of frozen CLIP / LoRA A+B / classifier head / optimizer / AMP
scaler / training args, model config, and total / trainable / frozen parameter
counts (authoritative via model-build requires_grad, cross-checked against the
raw state_dict).
"""
import argparse
import hashlib
import json
import os
import sys

# Force CPU: never initialize CUDA (eval may be running on a GPU).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

# Default repo = the directory that contains ForensicHub/ as a subdirectory.
# Resolved relative to this script so the check works regardless of the current
# working directory: <repo>/scripts/verify_checkpoint.py -> <repo>/.. (ForensicHub
# is a sibling of this repo under SICA_OpenMMSec). Override with --repo or set
# the FORENSICHUB env var to point directly at a ForensicHub checkout.
DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    sz = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            sz += len(b)
    return h.hexdigest(), sz


def classify_key(k):
    """Coarse classification of a state_dict key for presence checks."""
    kl = k.lower()
    if "parametrizations" in kl or "lora" in kl:
        return "lora"
    if k.startswith("fc.") or k.startswith("head.") or "classifier" in kl:
        return "head"
    if "visual" in kl or "model." in kl or k.startswith("model."):
        return "backbone"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Directory containing ForensicHub/ as a subdirectory "
             "(default: this repo's parent). Override with --repo, or set the "
             "FORENSICHUB env var to point directly at a ForensicHub checkout."
    )
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"[verify] FATAL: checkpoint not found: {ckpt_path}", flush=True)
        sys.exit(2)
    rec = {"absolute_path": ckpt_path}

    print(f"[verify] computing SHA-256 of {ckpt_path} ...", flush=True)
    sha, size = sha256_of(ckpt_path)
    rec["file_size_bytes"] = size
    rec["sha256"] = sha
    rec["sha256_prefix8"] = sha[:8]

    print("[verify] loading checkpoint on CPU (weights_only=False) ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("[verify] FATAL: checkpoint is not a dict", flush=True)
        sys.exit(2)

    top_keys = list(ckpt.keys())
    rec["top_level_keys"] = top_keys
    rec["epoch"] = ckpt.get("epoch", None)

    # ---- presence of non-model top-level entries ----------------------------
    rec["presence"] = {
        "optimizer": "optimizer" in ckpt,
        "amp_scaler": "scaler" in ckpt,
        "training_args": "args" in ckpt,
        "model_state": "model" in ckpt,
    }

    # ---- training args (model config) ---------------------------------------
    targs = ckpt.get("args", None)
    targs_dict = None
    if targs is not None:
        # args may be argparse Namespace -> dict
        if hasattr(targs, "__dict__"):
            targs_dict = {k: v for k, v in vars(targs).items()
                          if isinstance(v, (str, int, float, bool, type(None), list, tuple))}
        else:
            targs_dict = targs
    rec["training_args"] = targs_dict

    # ---- model state dict ----------------------------------------------------
    sd = ckpt["model"] if "model" in ckpt else ckpt
    rec["num_model_tensors"] = len(sd)

    # classify keys
    by_class = {"lora": [], "head": [], "backbone": [], "other": []}
    for k in sd.keys():
        by_class[classify_key(k)].append(k)
    rec["key_classes"] = {c: len(v) for c, v in by_class.items()}
    rec["lora_key_samples"] = sorted(by_class["lora"])[:6]
    rec["head_keys"] = sorted(by_class["head"])[:8]
    rec["backbone_key_samples"] = sorted(by_class["backbone"])[:4]

    # presence of LoRA A/B within the parametrization keys
    lora_keys = by_class["lora"]
    has_lora_A = any(("a" == k.rsplit(".", 1)[-1].lower()) or ".a" in k.lower() for k in lora_keys)
    has_lora_B = any(("b" == k.rsplit(".", 1)[-1].lower()) or ".b" in k.lower() for k in lora_keys)
    rec["presence"]["lora_params"] = len(lora_keys) > 0
    rec["presence"]["lora_A"] = has_lora_A
    rec["presence"]["lora_B"] = has_lora_B
    rec["presence"]["frozen_clip_backbone"] = len(by_class["backbone"]) > 0
    rec["presence"]["classifier_head"] = len(by_class["head"]) > 0

    # raw total param count from state dict
    total_from_sd = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
    rec["total_params_from_state_dict"] = total_from_sd

    # ---- authoritative trainable/frozen via model build ---------------------
    param_counts = {"method": "model_build_failed", "total": None,
                    "trainable": None, "frozen": None, "detail": None}
    try:
        sys.path.insert(0, os.environ.get("FORENSICHUB") or os.path.join(args.repo, "ForensicHub"))
        from ForensicHub.registry import MODELS, build_from_registry  # noqa: E402
        model_args = {"name": "CLIP_LORA_PURE", "init_config": {
            "name": "ViT-L/14", "num_classes": 1, "lora_r": 8, "lora_alpha": 16.0,
            "lora_qkv": "qkv", "lora_apply_out_proj": True, "tune_mode": "lora"}}
        print("[verify] building CLIP_LORA_PURE on CPU for requires_grad audit ...", flush=True)
        model = build_from_registry(MODELS, model_args)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        # cross-check: load state dict
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_lora_modules = getattr(model, "_lora_params", None)
        param_counts = {
            "method": "model_build_requires_grad",
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(frozen),
            "trainable_pct": round(100.0 * trainable / total, 4) if total else None,
            "missing_keys_on_load": len(missing),
            "unexpected_keys_on_load": len(unexpected),
            "num_lora_modules": len(n_lora_modules) if n_lora_modules is not None else None,
            "missing_key_samples": missing[:5],
            "unexpected_key_samples": unexpected[:5],
        }
    except Exception as e:
        param_counts["detail"] = f"{type(e).__name__}: {e}"
        print(f"[verify] model build failed ({e}); falling back to state_dict naming", flush=True)

    rec["parameter_counts"] = param_counts

    # ---- config summary (from args + init) ----------------------------------
    rec["model_config"] = {
        "backbone": "CLIP ViT-L/14",
        "lora_r": 8,
        "lora_alpha": 16.0,
        "lora_qkv": "qkv",
        "lora_apply_out_proj": True,
        "tune_mode": "lora",
        "num_classes": 1,
        "expected_num_lora_modules": 48,
        "classifier_head": "nn.Linear(768 -> 1) [self.fc]",
        "labels": {"0": "real", "1": "fake"},
    }

    # ---- write outputs ------------------------------------------------------
    with open(args.out_json, "w") as f:
        json.dump(rec, f, indent=2)

    md = []
    md.append("# SICA Checkpoint Verification")
    md.append("")
    md.append(f"- **Absolute path:** `{ckpt_path}`")
    md.append(f"- **File size:** {size:,} bytes ({size/1e9:.3f} GB)")
    md.append(f"- **SHA-256:** `{sha}`")
    md.append(f"- **SHA-256 (prefix):** `{sha[:8]}`")
    md.append(f"- **Epoch:** {rec['epoch']}")
    md.append(f"- **Top-level keys:** {top_keys}")
    md.append(f"- **Number of model tensors:** {rec['num_model_tensors']}")
    md.append("")
    md.append("## Presence checks")
    for k, v in rec["presence"].items():
        md.append(f"- **{k}:** {'YES' if v else 'NO'}")
    md.append("")
    md.append("## State-dict key classification (coarse, by name)")
    for c, n in rec["key_classes"].items():
        md.append(f"- {c}: {n} tensors")
    md.append(f"- LoRA key samples: `{rec['lora_key_samples']}`")
    md.append(f"- Head keys: `{rec['head_keys']}`")
    md.append("")
    md.append("## Parameter counts")
    pc = rec["parameter_counts"]
    md.append(f"- **Method:** {pc['method']}")
    md.append(f"- **Total parameters:** {pc.get('total')}")
    md.append(f"- **Trainable parameters:** {pc.get('trainable')} "
              f"({pc.get('trainable_pct')}%)")
    md.append(f"- **Frozen parameters:** {pc.get('frozen')}")
    md.append(f"- **Total from raw state_dict:** {total_from_sd:,}")
    if pc.get("num_lora_modules") is not None:
        md.append(f"- **LoRA modules registered:** {pc['num_lora_modules']} "
                  f"(expected 48)")
    md.append(f"- **Missing keys on load (strict=False):** {pc.get('missing_keys_on_load')}")
    md.append(f"- **Unexpected keys on load:** {pc.get('unexpected_keys_on_load')}")
    if pc.get("detail"):
        md.append(f"- **Detail:** {pc['detail']}")
    md.append("")
    md.append("## Model configuration")
    for k, v in rec["model_config"].items():
        md.append(f"- **{k}:** {v}")
    md.append("")
    md.append("## Training args (from checkpoint)")
    md.append("```json")
    md.append(json.dumps(rec.get("training_args") or {}, indent=2, default=str))
    md.append("```")
    md.append("")
    md.append("_Checkpoint file was NOT modified. CPU-only inspection._")

    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({k: rec[k] for k in
                      ["sha256", "file_size_bytes", "epoch", "top_level_keys",
                       "num_model_tensors", "parameter_counts"]}, indent=2),
          flush=True)
    print(f"[verify] wrote {args.out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()

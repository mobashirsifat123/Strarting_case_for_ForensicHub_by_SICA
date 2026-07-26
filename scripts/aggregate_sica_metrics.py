#!/usr/bin/env python3
"""aggregate_sica_metrics.py - compute paper-compatible metrics from a predictions CSV.

Reads a dump_sica_predictions.py CSV (path,label,probability,logit,pred_class,
domain,mani_type,sub_mani_type,ori_dataset) and computes:
  - global ACC/AUC/AP/F1
  - per-domain  (group by `domain`)
  - per-primary-faking-type (group by `mani_type`)
  - per-sub-mani-type (group by `sub_mani_type`)
  - macro-domain average = unweighted mean of per-domain metrics
  - macro-primary-type average = unweighted mean of per-mani_type metrics

Metric implementations MATCH the paper's codebase exactly:
  AUC = sklearn.roc_auc_score(label, probability)         [IMDLBenCo ImageAUC]
  AP  = sklearn.average_precision_score(label, probability) [== ForensicHub average_precision_gpu]
  ACC = accuracy at threshold 0.5 on probability           [IMDLBenCo ImageAccuracy, threshold=0.5]
  F1  = sklearn.f1_score(label, pred_class) at threshold 0.5 [IMDLBenCo ImageF1, threshold=0.5]
label semantics: 0 = real, 1 = fake (pred_class = 1 if probability >= threshold).

USAGE:
  python aggregate_sica_metrics.py --input preds.csv --out-json metrics.json --out-md metrics.md \
      [--threshold 0.5] [--split-name total_ood_test]
"""
import argparse, csv, json, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

THRESHOLD_DEFAULT = 0.5

def metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    out = {"count": int(len(labels)),
           "n_real": int(np.sum(labels == 0)),
           "n_fake": int(np.sum(labels == 1))}
    out["ACC"] = float(accuracy_score(labels, pred)) if len(labels) else None
    out["F1"] = float(f1_score(labels, pred, zero_division=0)) if len(labels) else None
    if len(np.unique(labels)) >= 2:
        out["AUC"] = float(roc_auc_score(labels, probs))
        out["AP"] = float(average_precision_score(labels, probs))
    else:
        out["AUC"] = None
        out["AP"] = None
        out["single_class_warning"] = "only one class present; AUC/AP undefined"
    return out

def macro(vals, key):
    xs = [v[key] for v in vals if v.get(key) is not None]
    return float(np.mean(xs)) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--split-name", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"]); r["probability"] = float(r["probability"])
            rows.append(r)
    if not rows:
        print("ERROR: no rows in CSV", file=sys.stderr); sys.exit(2)
    labels = [r["label"] for r in rows]
    probs = [r["probability"] for r in rows]

    g = metrics(labels, probs, args.threshold)

    def group_by(key):
        d = defaultdict(list)
        for r in rows:
            d[r.get(key, "") or "<unspecified>"].append(r)
        return {k: metrics([x["label"] for x in v], [x["probability"] for x in v], args.threshold)
                for k, v in sorted(d.items())}

    per_domain = group_by("domain")
    per_mani = group_by("mani_type")
    per_sub = group_by("sub_mani_type")

    result = {
        "split": args.split_name or args.input,
        "threshold": args.threshold,
        "global": g,
        "per_domain": per_domain,
        "per_primary_type": per_mani,
        "per_sub_mani_type": per_sub,
        "macro_domain_average": {k: macro(per_domain.values(), k) for k in ["ACC", "AUC", "AP", "F1"]},
        "macro_primary_type_average": {k: macro(per_mani.values(), k) for k in ["ACC", "AUC", "AP", "F1"]},
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    if args.out_md:
        write_md(result, args.out_md)
    # stdout summary
    print(json.dumps({"split": result["split"], "global": g,
                      "macro_domain": result["macro_domain_average"]}, indent=2))
    return result

def write_md(r, path):
    L = [f"# Metrics: {r['split']}", "", f"threshold = {r['threshold']}", "",
         "## Global", "| ACC | AUC | AP | F1 | count | real | fake |",
         "|---|---|---|---|---|---|---|"]
    g = r["global"]
    L.append(f"| {g['ACC']:.6f} | {fmt(g['AUC'])} | {fmt(g['AP'])} | {fmt(g['F1'])} | {g['count']} | {g['n_real']} | {g['n_fake']} |")
    L += ["", "## Macro-domain average (unweighted mean of per-domain)",
          "| ACC | AUC | AP | F1 |", "|---|---|---|---|"]
    m = r["macro_domain_average"]
    L.append(f"| {fmt(m['ACC'])} | {fmt(m['AUC'])} | {fmt(m['AP'])} | {fmt(m['F1'])} |")
    L += ["", "## Per-domain", "| domain | count | ACC | AUC | AP | F1 |", "|---|---|---|---|---|---|"]
    for k, v in r["per_domain"].items():
        L.append(f"| {k} | {v['count']} | {fmt(v['ACC'])} | {fmt(v['AUC'])} | {fmt(v['AP'])} | {fmt(v['F1'])} |")
    L += ["", "## Per-primary-type (mani_type)", "| type | count | ACC | AUC | AP | F1 |", "|---|---|---|---|---|---|"]
    for k, v in r["per_primary_type"].items():
        L.append(f"| {k} | {v['count']} | {fmt(v['ACC'])} | {fmt(v['AUC'])} | {fmt(v['AP'])} | {fmt(v['F1'])} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

def fmt(x):
    return f"{x:.6f}" if isinstance(x, (int, float)) and x is not None else "undefined"

if __name__ == "__main__":
    main()

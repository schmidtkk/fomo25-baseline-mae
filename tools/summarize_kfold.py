#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

from utils.ensemble import average_subject_probabilities, auroc_from_subject_probabilities


def find_latest_subject_json(fold_dir: str) -> str | None:
    # Expect structure: <base>/foldX/Task001_FOMO1/<model>/version_*/subject_probs/val_subject_probs_epoch_XXXX.json
    cand_dirs: List[str] = []
    for root, dirs, files in os.walk(fold_dir):
        if os.path.basename(root) == "subject_probs":
            cand_dirs.append(root)
    if not cand_dirs:
        return None
    latest_path = None
    latest_epoch = -1
    pat = re.compile(r"val_subject_probs_epoch_(\d+)\.json$")
    for sp_dir in cand_dirs:
        for fname in os.listdir(sp_dir):
            m = pat.match(fname)
            if m:
                ep = int(m.group(1))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_path = os.path.join(sp_dir, fname)
    return latest_path


def load_subject_json(path: str) -> Tuple[Dict[str, float], Dict[str, int] | None]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("subject_probs", {}), data.get("subject_targets", None)


def main():
    parser = argparse.ArgumentParser(description="Summarize K-fold subject-level AUROC and ensemble")
    parser.add_argument("base_dir", help="Base directory containing per-fold subdirectories (e.g., runs/fomo1_k3)")
    parser.add_argument("--fold_prefix", default="fold", help="Prefix for fold folders (default: fold)")
    parser.add_argument("--num_folds", type=int, required=True, help="Number of folds to summarize")
    parser.add_argument("--print_paths", action="store_true", help="Print located JSON paths per fold")
    args = parser.parse_args()

    subject_probs_list: List[Dict[str, float]] = []
    subject_targets: Dict[str, int] | None = None
    per_fold_auc: List[float] = []

    for f in range(args.num_folds):
        fold_dir = os.path.join(args.base_dir, f"{args.fold_prefix}{f}")
        json_path = find_latest_subject_json(fold_dir)
        if json_path is None:
            print(f"[WARN] No subject_probs JSON found for fold {f} under {fold_dir}")
            continue
        if args.print_paths:
            print(f"Fold {f}: {json_path}")
        probs, targets = load_subject_json(json_path)
        subject_probs_list.append(probs)
        if targets is not None:
            subject_targets = targets
            try:
                auc = auroc_from_subject_probabilities(probs, targets)
            except Exception:
                auc = float("nan")
            per_fold_auc.append(auc)

    if len(subject_probs_list) == 0:
        print("No folds with subject probabilities found. Ensure --export_subject_probs was enabled during training.")
        return

    print("Per-fold AUROC (if targets available):")
    if per_fold_auc:
        for i, auc in enumerate(per_fold_auc):
            print(f"  Fold {i}: {auc:.6f}")
        mean_auc = sum(x for x in per_fold_auc if x == x) / max(1, len(per_fold_auc))
        print(f"  Mean: {mean_auc:.6f}")
    else:
        print("  (targets missing; cannot compute per-fold AUROC)")

    # Ensemble across folds
    averaged = average_subject_probabilities(subject_probs_list, weights=None)
    if subject_targets is not None:
        ens_auc = auroc_from_subject_probabilities(averaged, subject_targets)
        print(f"Ensembled AUROC across {len(subject_probs_list)} folds: {ens_auc:.6f}")
    else:
        print("Ensembled subject probabilities computed, but targets missing; AUROC unavailable.")


if __name__ == "__main__":
    main()



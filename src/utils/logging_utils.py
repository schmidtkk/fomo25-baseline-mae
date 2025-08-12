from __future__ import annotations

import csv
import os
from typing import Dict


def write_subject_csv(path: str, subject_probs: Dict[str, float], subject_targets: Dict[str, int], epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "prob_pos", "target", "epoch"])
        for sid in sorted(subject_probs.keys()):
            prob = subject_probs[sid]
            tgt = subject_targets.get(sid, "")
            w.writerow([sid, prob, tgt, epoch])



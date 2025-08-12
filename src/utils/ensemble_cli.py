import argparse
import json
from typing import List

from utils.ensemble import average_subject_probabilities, auroc_from_subject_probabilities


def load_subject_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    subject_probs = data.get("subject_probs", {})
    subject_targets = data.get("subject_targets", None)
    return subject_probs, subject_targets


def main(args):
    subject_probs_list: List[dict] = []
    subject_targets = None
    for inp in args.inputs:
        probs, targets = load_subject_json(inp)
        subject_probs_list.append(probs)
        if targets is not None:
            subject_targets = targets

    averaged = average_subject_probabilities(subject_probs_list, weights=args.weights)

    out = {"subject_probs": averaged}
    if subject_targets is not None:
        out["subject_targets"] = subject_targets
        auroc_val = auroc_from_subject_probabilities(averaged, subject_targets)
        print(f"Subject-level AUROC (ensemble): {auroc_val}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote ensemble subject probabilities to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble per-subject probabilities JSONs")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of input JSON files")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights per input, same order as --inputs",
    )
    args = parser.parse_args()
    main(args)



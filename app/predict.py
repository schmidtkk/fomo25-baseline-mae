#!/usr/bin/env python
import argparse
import os
import json

import nibabel as nib
import numpy as np
import torch

from inference.predict_utils import (
    average_scalar_probabilities,
    parse_comma_separated_floats,
    apply_temperature_to_prob,
)


def load_nifti(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    return arr


def preprocess_modalities(flair: str, adc: str, dwi_b1000: str, t2s: str | None, swi: str | None) -> torch.Tensor:
    vols = []
    for p in [dwi_b1000, adc, flair, (swi or t2s)]:
        if p is None:
            vols.append(None)
        else:
            vols.append(load_nifti(p))
    # Basic sanity: all volumes must have same shape
    shapes = [v.shape for v in vols if v is not None]
    assert len(shapes) > 0 and all(s == shapes[0] for s in shapes), "Input modalities must share shape"
    D, H, W = shapes[0]
    stacked = np.zeros((4, D, H, W), dtype=np.float32)
    for i, v in enumerate(vols):
        if v is not None:
            stacked[i] = v
    # Add batch dimension
    return torch.from_numpy(stacked[None, ...])


def build_model(checkpoint: str, device: torch.device):
    # Lazy import to avoid heavy deps in container init
    from models.supervised_base import BaseSupervisedModel
    # Minimal config to instantiate model; must match training
    # NOTE: Adjust these according to your trained model setup before building container
    config = {
        "num_classes": 2,
        "num_modalities": 4,
        "patch_size": (32, 32, 32),
        "model_name": "unet_xl",
        "version_dir": ".",
        "task_type": "classification",
    }
    model = BaseSupervisedModel.create(task_type="classification", config=config)
    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def predict_proba(model, volume: torch.Tensor, device: torch.device, tta: bool = False) -> float:
    # Simple center-crop to model patch size if needed; here we assume model supports sliding window internally if integrated
    volume = volume.to(device)
    logits = model(volume)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    if tta:
        # Optional simple flips at inference; average probabilities
        probs = [prob]
        vol = volume
        for dims in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
            flipped = torch.flip(vol, dims=dims)
            p = torch.softmax(model(flipped), dim=1)[0, 1].item()
            probs.append(p)
        prob = float(np.mean(probs))
    return float(prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flair", required=True)
    parser.add_argument("--adc", required=True)
    parser.add_argument("--dwi_b1000", required=True)
    parser.add_argument("--t2s", default=None)
    parser.add_argument("--swi", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT", ""))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable_tta", action="store_true")
    parser.add_argument("--calibration_json", default=None, help="JSON with temperature value {\"temperature\": T}")
    parser.add_argument("--fold_checkpoints", default=None, help="Comma-separated checkpoints to ensemble")
    parser.add_argument("--fold_weights", default=None, help="Optional comma-separated weights for fold ensembling")
    args = parser.parse_args()

    device = torch.device(args.device)
    vol = preprocess_modalities(args.flair, args.adc, args.dwi_b1000, args.t2s, args.swi)

    # Single checkpoint or fold ensemble
    probs = []
    if args.fold_checkpoints:
        ckpts = [p.strip() for p in args.fold_checkpoints.split(",") if p.strip()]
        for ck in ckpts:
            model = build_model(ck, device)
            p = predict_proba(model, vol, device, tta=bool(args.enable_tta))
            probs.append(p)
        weights = parse_comma_separated_floats(args.fold_weights)
        prob = average_scalar_probabilities(probs, weights)
    else:
        model = build_model(args.checkpoint, device)
        prob = predict_proba(model, vol, device, tta=bool(args.enable_tta))

    # Optional temperature scaling at inference
    if args.calibration_json and os.path.isfile(args.calibration_json):
        with open(args.calibration_json, "r") as f:
            data = json.load(f)
        T = float(data.get("temperature", 1.0))
        prob = apply_temperature_to_prob(prob, T)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"{prob:.6f}\n")


if __name__ == "__main__":
    main()



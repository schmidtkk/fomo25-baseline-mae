## Task 1 Inference (Infarct Detection)

### Container layout (required)
/
├── app/
│   └── predict.py   # entrypoint (REQUIRED)
├── input/           # mounted at runtime (do not bake in)
├── output/          # mounted at runtime (do not bake in)
└── ...

### Provided in this repo
- `app/predict.py`: CLI
  - Required args: `--flair`, `--adc`, `--dwi_b1000`, optional `--t2s/--swi`, and `--output` (writes a single probability as text).
  - Optional: `--checkpoint` (path inside container), `--device`, `--enable_tta`, `--calibration_json` (with `{ "temperature": T }`).
- `Apptainer.def`: builds an Apptainer image based on `python:3.11-slim`, installs requirements, sets runscript to `/app/predict.py`.
- `src/inference/container_requirements.txt`: minimal dependency list.

### Predict usage
```bash
python /app/predict.py \
  --flair /input/flair.nii.gz \
  --adc /input/adc.nii.gz \
  --dwi_b1000 /input/dwi_b1000.nii.gz \
  --swi /input/swi.nii.gz \
  --output /output/prob.txt \
  --checkpoint /workspace/checkpoints/best.ckpt
```

### Notes
- Adjust the `build_model` config in `app/predict.py` to match your trained model (modalities, patch size, `model_name`).
- TTA is off by default and only recommended if similar flips were used in training (we do not by default).
- Temperature scaling can be applied via `--calibration_json` using a temperature learned on validation data with `utils.calibration.TemperatureScaler`.



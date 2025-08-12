#!/usr/bin/env bash

MODALITY="${1:-t1}"

echo "[RUN] Launching pretraining"
echo "[RUN] Modality: ${MODALITY}"
echo "[RUN] Timestamp: $(date)"

sh pretrain.sh "$MODALITY"
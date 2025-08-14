PYTHONPATH=src python src/finetune.py \
  --taskid 1 \
  --data_dir /mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-finetune \
  --save_dir ./runs \
  --model_name unet_xl \
  --fusion_mode fusion \
  --dwi_ckpt   /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/dwi.ckpt \
  --flair_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/flair.ckpt \
  --t1_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t1.ckpt \
  --t2_ckpt    /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/t2.ckpt \
  --other_ckpt /mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main/ckpt/other.ckpt \
  --precision 16-mixed \
  --patch_size=96 \
  --train_batches_per_epoch=100 \
  --epochs 500 --batch_size 2 --val_batch_size 8 --accumulate_grad_batches 1 --num_devices 1 --num_workers 8 --new_version \
  --starting_filters 64 \
  --freeze_encoder_epochs 15 --phase1_head_lr 1e-4 --phase2_head_lr 1e-4 --phase2_encoder_lr 2e-5 \
  --label_smoothing 0.05 --cls_head_dropout_p 0.05 \
  --grad_clip_val 1.0 --grad_clip_algo norm \
  --num_sanity_val_steps 0 --log_every_n_steps 50 \
  --disable_early_stop \
  --augmentation_preset basic
  # --lr_scheduler plateau --plateau_factor 0.5 --plateau_patience 6 --plateau_threshold 1e-3 --plateau_cooldown 0 --plateau_min_lr 1e-7 \
  # --val_batch_size 8 \
  # --val_tta_enable --val_tta_views 8 \
  # --val_tta_offsets 7 --val_tta_offset_frac 0.25 \
  # --val_tta_batch_size 8 \
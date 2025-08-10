DATA_DIR=/mnt/cvlab/scratch/cvlab/home/hantzhan/data/FOMO-MRI/fomo-60k-pretrain/FOMO60k
SAVE_DIR=/mnt/cvlab/scratch/cvlab/home/hantzhan/code/fomo25-baseline-mae-main
MODALITY=t1

CUDA_VISIBLE_DEVICES=0 python src/pretrain.py \
    --save_dir=$SAVE_DIR \
    --pretrain_data_dir=$DATA_DIR \
    --model_name=unet_xl_lw_dec \
    --patch_size=128 \
    --batch_size=16 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=4 \
    --augmentation_preset=all \
    --num_devices 1 \
    --accumulate_grad_batches 1 \
    --modality_mode $MODALITY \
    --experiment $MODALITY \
    --skip_sanity_check \
    --prefetch_factor=16 \
    --disable_memmap


# python src/pretrain.py \
#     --save_dir=/data/weidong/models \
#     --pretrain_data_dir=/home/weidongguo/workspace/fomo2025/data \
#     --model_name=unet_b_lw_dec \
#     --patch_size=96 \
#     --batch_size=4 \
#     --epochs=100 \
#     --warmup_epochs=5 \
#     --num_workers=1 \
#     --augmentation_preset=all
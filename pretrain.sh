CUDA_VISIBLE_DEVICES=1 python src/pretrain.py \
    --save_dir=/data/weidong/models \
    --pretrain_data_dir=/data/weidong/fomo60k_preprocessed/FOMO60k \
    --model_name=unet_xl_lw_dec \
    --patch_size=128 \
    --batch_size=8 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=28 \
    --augmentation_preset=all \
    --num_devices 1 \
    --accumulate_grad_batches 1


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
# finetune 

# CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
#     --data_dir=data/preprocessed/ \
#     --save_dir=/data/weidong/models \
#     --pretrained_weights_path=/data/weidong/models/models/FOMO60k/unet_xl_lw_dec/versions/version_0/epoch=99.ckpt \
#     --model_name=unet_xl \
#     --patch_size=128 \
#     --taskid=1 \
#     --batch_size=2 \
#     --epochs=500 \
#     --train_batches_per_epoch=100 \
#     --augmentation_preset=all \
#     --new_version


# from scrach

CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
    --data_dir=data/preprocessed/ \
    --save_dir=/data/weidong/models \
    --model_name=unet_xl \
    --patch_size=128 \
    --taskid=4 \
    --batch_size=2 \
    --epochs=500 \
    --train_batches_per_epoch=100 \
    --augmentation_preset=all \
    --new_version



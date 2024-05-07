
python main.py \
    --data_root /storage/brno2/home/map/AdaFace/data/faces_webface_112x112 \
    --train_data_path train \
    --val_data_path . \
    --prefix ir50_ms1mv2_adaface \
    --use_mxrecord \
    --gpus 1 \
    --use_16bit \
    --arch ir_50 \
    --batch_size 512 \
    --num_workers 16 \
    --epochs 26 \
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2 \
    --resume_from_checkpoint ./pretrained/adaface_ir50_ms1mv2.ckpt



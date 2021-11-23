
python test.py --name mit5k-default \
    --dataset_mode mit5k \
    --nThreads 8 \
    --tileSize 256  \
    --gpu_ids 0 \
    --load_size 1024 \
    --crop_size 1024 \
    --preprocess_mode "none" \
    --phase test \
    --netG ConditionedUnet \
    --base_channels 32 \
    --global_base_channels 16 \
    --condition_hidden_channels 128 \
    --norm_groups 16 \
    --which_epoch best-psnr \
    --condition preprocess
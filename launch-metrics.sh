# Mit 5k PSNR & SSIM
python metrics.py --gpu 0 -e mit5k-default --tileSize 256 --step best-psnr

# Facades
#python fid_score.py ./results/facades-default/test_tile-256_crop-512_latest/images/gt ./results/facades-default/test_tile-256_crop-512_latest/images/synthesized_image --gpu 0 --batch-size 1

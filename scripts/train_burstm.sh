# Please modify the path of input directory
# bash ./scripts/train_burstm.sh
# CUDA_VISIBLE_DEVICES=0, python BurstM_Track_1_training.py \
#                     --input_dir D:/Vinh/3.Project_working/camera_isp/Datasets/Zurich-RAW-to-DSLR-Dataset \
#                     --log_dir ./../logs/burstM \
#                     --model_dir ./../logs/burstM/checkpoints \
#                     --result_dir ./../logs/burstM/results \
#                     --weights ./../logs/burstM/checkpoints/epoch=00-val_psnr=30.61.ckpt

CUDA_VISIBLE_DEVICES=0, python BurstM_Track_1_training.py \
                    --input_dir D:/Vinh/3.Project_working/camera_isp/Datasets/Zurich-RAW-to-DSLR-Dataset \
                    --log_dir ./../logs/latentburst \
                    --model_dir ./../logs/latentburst/checkpoints \
                    --result_dir ./../logs/latentburst/results \
                    --scale=4  \
                    # --weights ./../logs/burstM/checkpoints/epoch=04-val_psnr=34.44.ckpt
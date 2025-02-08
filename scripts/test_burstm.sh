# bash ./scripts/test_burstm.sh


# Please modify the path of iamge directory for inputs and pre-trained models(weights).
# CUDA_VISIBLE_DEVICES=0 python BurstM_Track_1_evaluation.py \
#                     --input_dir D:/Vinh/3.Project_working/camera_isp/Datasets/Zurich-RAW-to-DSLR-Dataset \
#                     --scale=4  \
#                     --weights D:/Vinh/3.Project_working/BurstSR/logs/burstM/pretrained/epoch=294-val_psnr=42.87.ckpt \
#                     --result_dir ./../logs/burstM/results/sythesis \
#                     --result_gt_dir ./../logs/burstM/results/gt

CUDA_VISIBLE_DEVICES=0 python BurstM_Track_1_evaluation.py \
                    --input_dir D:/Vinh/3.Project_working/camera_isp/Datasets/Zurich-RAW-to-DSLR-Dataset \
                    --scale=2  \
                    --weights ./../logs/burstM/checkpoints/epoch=04-val_psnr=34.44.ckpt \
                    --result_dir ./../logs/burstM/results_latentburst/sythesis \
                    --result_gt_dir ./../logs/burstM/results_latentburst/gt
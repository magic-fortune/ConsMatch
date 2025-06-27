#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='acdc'
method='ConsMatch'
exp='unet'

note='surp_add_loss_'
config=configs/$dataset.yaml


# etas=('0.5')
# # Assuming split is now an array
# splits=('3' '7') 

# for split in "${splits[@]}"
# do
#     for eta in "${etas[@]}"
#     do
#         note=surp_add_loss_kl_beishu_$eta
#         labeled_id_path=splits/$dataset/$split/labeled.txt
#         unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
#         save_path=exp/$dataset/${method}_$note/$exp/${split}_mse/eta_$eta
#         mkdir -p $save_path
#         OMP_NUM_THREADS=1 torchrun \
#             --nproc_per_node=$1 \
#             --master_addr=localhost \
#             --master_port=$2 \
#             $method.py \
#             --config=$config \
#             --labeled-id-path \
#             $labeled_id_path \
#             --unlabeled-id-path \
#             $unlabeled_id_path \
#             --save-path $save_path \
#             --port $2 \
#             --s1_to_s2 \
#             --beishu $eta \
#             --corr_match_type 'mse' \
#             --temperature 1.5 \
#             --eta $eta 2>&1 | tee $save_path/$now.log
#     done
# done





# etas=('0.5')
# # Assuming split is now an array
# splits=('3' '7') 

# for split in "${splits[@]}"
# do
#     for eta in "${etas[@]}"
#     do
#         note=surp_add_loss_kl_beishu_$eta
#         labeled_id_path=splits/$dataset/$split/labeled.txt
#         unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
#         save_path=exp/$dataset/${method}_$note/$exp/${split}_no_s1_s2/eta_$eta
#         mkdir -p $save_path
#         OMP_NUM_THREADS=1 torchrun \
#             --nproc_per_node=$1 \
#             --master_addr=localhost \
#             --master_port=$2 \
#             $method.py \
#             --config=$config \
#             --labeled-id-path \
#             $labeled_id_path \
#             --unlabeled-id-path \
#             $unlabeled_id_path \
#             --save-path $save_path \
#             --port $2 \
#             --beishu $eta \
#             --corr_match_type 'kl' \
#             --temperature 1.5 \
#             --eta $eta 2>&1 | tee $save_path/$now.log
#     done
# done





etas=('0.5')
# Assuming split is now an array
splits=('3' '7') 

for split in "${splits[@]}"
do
    for eta in "${etas[@]}"
    do
        note=surp_add_loss_kl_beishu_$eta
        labeled_id_path=splits/$dataset/$split/labeled.txt
        unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
        save_path=exp/$dataset/${method}_$note/$exp/${split}_only_corr_mt/eta_$eta
        mkdir -p $save_path
        OMP_NUM_THREADS=1 torchrun \
            --nproc_per_node=$1 \
            --master_addr=localhost \
            --master_port=$2 \
            $method.py \
            --config=$config \
            --labeled-id-path \
            $labeled_id_path \
            --unlabeled-id-path \
            $unlabeled_id_path \
            --save-path $save_path \
            --port $2 \
            --beishu $eta \
            --s1_to_s2 \
            --corr_match_type 'kl' \
            --only-corr-mt True\
            --temperature 1.5 \
            --eta $eta 2>&1 | tee $save_path/$now.log
    done
done




# etas=('0.5')
# # Assuming split is now an array
# splits=('3' '7') 

# for split in "${splits[@]}"
# do
#     for eta in "${etas[@]}"
#     do
#         note=surp_add_loss_kl_beishu_$eta
#         labeled_id_path=splits/$dataset/$split/labeled.txt
#         unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
#         save_path=exp/$dataset/${method}_$note/$exp/${split}_only_surp_add/eta_$eta
#         mkdir -p $save_path
#         OMP_NUM_THREADS=1 torchrun \
#             --nproc_per_node=$1 \
#             --master_addr=localhost \
#             --master_port=$2 \
#             $method.py \
#             --config=$config \
#             --labeled-id-path \
#             $labeled_id_path \
#             --unlabeled-id-path \
#             $unlabeled_id_path \
#             --save-path $save_path \
#             --port $2 \
#             --beishu $eta \
#             --s1_to_s2 \
#             --corr_match_type 'kl' \
#             --only-surp-add True\
#             --temperature 1.5 \
#             --eta $eta 2>&1 | tee $save_path/$now.log
#     done
# done

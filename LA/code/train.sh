#!/bin/bash
# BEST: Who Can Win?  s1_to_s2 or not ?
# python train.py \
# --exp "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2" \
# --conf_thresh 0.75 \
# --label_num 8 \
# --max_iterations 8000 \
# --optimizer AdamW \
# --base_lr 0.005 \
# --eta 0.5 \
# --s1_to_s2

# python train.py \
# --exp "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/no_s1_to_s2" \
# --conf_thresh 0.75 \
# --label_num 8 \
# --max_iterations 8000 \
# --optimizer AdamW \
# --base_lr 0.005 \
# --eta 0.5

# for eta in $(seq 0.0 0.1 1.0)
# do
#     if [ $eta == 0.5 ]
#     then
#         continue
#     fi
#     python train.py \
#     --exp "lab8/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" \
#     --conf_thresh 0.75 \
#     --label_num 8 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.005 \
#     --eta ${eta} \
#     --s1_to_s2
# done

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python train.py \
#     --exp "lab4/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" \
#     --conf_thresh 0.75 \
#     --label_num 4 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.005 \
#     --eta ${eta} \
#     --s1_to_s2
# done

# for tau in $(seq 0.71 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python train.py \
#     --exp "lab8/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" \
#     --conf_thresh ${tau} \
#     --label_num 8 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.005 \
#     --eta 0.5 \
#     --s1_to_s2
# done

# for tau in $(seq 0.70 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python test.py --model "lab8/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" --epoch_num 7000
# done


# for tau in $(seq 0.70 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python train.py \
#     --exp "lab16/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" \
#     --conf_thresh ${tau} \
#     --label_num 16 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.001 \
#     --eta 0.5 \
#     --s1_to_s2
# done

# for tau in $(seq 0.70 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python train.py \
#     --exp "lab4/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" \
#     --conf_thresh ${tau} \
#     --label_num 4 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.001 \
#     --eta 0.5 \
#     --s1_to_s2
# done

# for tau in $(seq 0.70 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python test.py --model "lab16/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" --epoch_num 8001
# done

# for tau in $(seq 0.70 0.01 0.80)
# do
#     if [ $tau == 0.75 ]
#     then
#         continue
#     fi
#     python test.py --model "lab4/AdamW/8k/e0.5/t${tau}/no_cutmix/s1_to_s2" --epoch_num 8001
# done

# python train.py \
# --exp "lab8/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2/alpdrop${pert_ratio}" \
# --conf_thresh 0.75 \
# --label_num 8 \
# --max_iterations 8000 \
# --optimizer AdamW \
# --base_lr 0.001 \
# --eta 0.5 \
# --s1_to_s2

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python train.py \
#     --exp "lab16/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" \
#     --conf_thresh 0.75 \
#     --label_num 16 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.001 \
#     --eta ${eta} \
#     --s1_to_s2
# done

# python test.py --model "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2" --epoch_num 8001
# python test.py --model "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/no_s1_to_s2" --epoch_num 8001

# for eta in $(seq 0.0 0.1 1.0)
# do
#     if [ $eta == 0.5 ]
#     then
#         continue
#     fi
#     python test.py --model "lab8/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" --epoch_num 8001
# done

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python test.py --model "lab4/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" --epoch_num 8001
# done

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python test.py --model "lab16/AdamW/8k/e${eta}/t0.75/no_cutmix/s1_to_s2" --epoch_num 7000
# done

# # for eta in $(seq 0.0 0.1 1.0)
# # do
# #     python test.py --model "AdamW_8000_e${eta}_lab8" --epoch_num 8001
# # done

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python test.py --model "AdamW_8000_e${eta}_lab16" --epoch_num 8001
# done

# for eta in $(seq 0.4 0.1 0.6)
# do
#     python test.py --model "AdamW_8000_e${eta}_lab4" --epoch_num 8001
# done

# for T in $(seq 1 0.5 4)
# do
#     python train.py \
#     --exp "lab16/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl${T}_0.005" \
#     --conf_thresh 0.75 \
#     --label_num 16 \
#     --max_iterations 8000 \
#     --optimizer AdamW \
#     --base_lr 0.005 \
#     --eta 0.5 \
#     --s1_to_s2 \
#     --corr_match_type 'kl' \
#     --temperature ${T}
# done

# for T in $(seq 1 0.5 4)
# do
#     python test.py --model "lab16/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl${T}_0.005" --epoch_num 7000
# done


# python train.py \
# --exp "lab16/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" \
# --conf_thresh 0.75 \
# --label_num 16 \
# --max_iterations 16000 \
# --optimizer AdamW \
# --base_lr 0.001 \
# --eta 0.5 \
# --s1_to_s2 \
# --corr_match_type 'kl' \
# --temperature 1.5

# python train.py \
# --exp "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" \
# --conf_thresh 0.75 \
# --label_num 8 \
# --max_iterations 16000 \
# --optimizer AdamW \
# --base_lr 0.001 \
# --eta 0.5 \
# --s1_to_s2 \
# --corr_match_type 'kl' \
# --temperature 1.5

# ICME 25 Re run!!!!!!
# python train.py \
# --exp "lab4/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" \
# --conf_thresh 0.75 \
# --label_num 4 \
# --max_iterations 38000 \
# --optimizer AdamW \
# --base_lr 0.001 \
# --eta 0.5 \
# --s1_to_s2 \
# --corr_match_type 'kl' \
# --temperature 1.5

# python train.py \
# --exp "ConsMatch/lab4/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/" \
# --conf_thresh 0.75 \
# --label_num 4 \
# --max_iterations 38000 \
# --optimizer AdamW \
# --base_lr 0.001 \
# --beishu 0.5 \
# --s1_to_s2 \
# --corr_match_type 'kl' \
# --temperature 1.5

python ConsMatch.py \
--exp "ConsMatch_now/lab8/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/" \
--conf_thresh 0.75 \
--label_num 8 \
--max_iterations 38000 \
--optimizer AdamW \
--base_lr 0.001 \
--eta 0.5 \
--s1_to_s2 \
--corr_match_type 'kl' \
--temperature 1.5


python ConsMatch.py \
--exp "ConsMatch_now/lab16/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/" \
--conf_thresh 0.75 \
--label_num 16 \
--max_iterations 38000 \
--optimizer AdamW \
--base_lr 0.001 \
--eta 0.5 \
--s1_to_s2 \
--corr_match_type 'kl' \
--temperature 1.5

# python test.py --model "lab16/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" --epoch_num best_model
# python test.py --model "lab8/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" --epoch_num best_model
# python test.py --model "lab4/AdamW/8k/e0.5/t0.75/no_cutmix/s1_to_s2/kl1.5_0.005" --epoch_num best_model
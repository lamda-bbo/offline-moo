# python config_evoxbench.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
for env_name in "regex"; do
    for seed in 1; do
        for train_data_mode in "none"; do
            for train_mode in "none"; do
                for output_size in 256; do
                    python config_evoxbench.py &&  \
                    CUDA_VISIBLE_DEVICES=0 \
                    python scripts/multi_obj_nn.py \
                    --env-name ${env_name} \
                    --seed ${seed} \
                    --normalize-y \
                    --num-solutions ${output_size} \
                    --filter-type best \
                    --train-data-mode ${train_data_mode} \
                    --train-mode ${train_mode} \
                    --reweight-mode none \
                    --df-name "test-test-hv.csv"
                done
            done
        done
    done
done

# for env_name in "motsp_100" "motsp_50" "motsp_20"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for output_size in 256; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_obj_nn.py \
#                     --env-name ${env_name} \
#                     --seed ${seed} \
#                     --normalize-y \
#                     --num-solutions ${output_size} \
#                     --filter-type best \
#                     --train-data-mode ${train_data_mode} \
#                     --train-mode ${train_mode} \
#                     --reweight-mode none \
#                     --df-name "1-test-hv.csv"
#                 done
#             done
#         done
#     done
# done

# for env_name in "motsp_100" "motsp_50" "motsp_20"; do
#     for seed in 1; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "pcgrad" "grad_norm"; do
#                 for output_size in 256 32; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_obj_nn.py \
#                     --env-name ${env_name} \
#                     --seed ${seed} \
#                     --normalize-y \
#                     --num-solutions ${output_size} \
#                     --filter-type best \
#                     --train-data-mode ${train_data_mode} \
#                     --train-mode ${train_mode} \
#                     --reweight-mode none \
#                     --df-name "1-test-hv.csv"
#                 done
#             done
#         done
#     done
# done

# for env_name in "motsp_100" "motsp_50" "motsp_20"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for output_size in 256 32; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_obj_nn.py \
#                     --env-name ${env_name} \
#                     --seed ${seed} \
#                     --normalize-y \
#                     --num-solutions ${output_size} \
#                     --filter-type best \
#                     --train-data-mode ${train_data_mode} \
#                     --train-mode ${train_mode} \
#                     --reweight-mode none \
#                     --df-name "1-test-hv.csv"
#                 done
#             done
#         done
#     done
# done

# for env_name in "motsp3obj_100" "motsp3obj_50" "motsp3obj_20"; do
#     for seed in 2; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "pcgrad" "grad_norm"; do
#                 for output_size in 256 32; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_obj_nn.py \
#                     --env-name ${env_name} \
#                     --seed ${seed} \
#                     --normalize-y \
#                     --num-solutions ${output_size} \
#                     --filter-type best \
#                     --train-data-mode ${train_data_mode} \
#                     --train-mode ${train_mode} \
#                     --reweight-mode none \
#                     --df-name "2-test-hv.csv"
#                 done
#             done
#         done
#     done
# done

# for env_name in "motsp_100" "motsp_50" "motsp_20"; do
#     for seed in 2; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for output_size in 256 32; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_obj_nn.py \
#                     --env-name ${env_name} \
#                     --seed ${seed} \
#                     --normalize-y \
#                     --num-solutions ${output_size} \
#                     --filter-type best \
#                     --train-data-mode ${train_data_mode} \
#                     --train-mode ${train_mode} \
#                     --reweight-mode none \
#                     --df-name "2-test-hv.csv"
#                 done
#             done
#         done
#     done
# done



# "c10mop1" "c10mop2" "c10mop3" "c10mop4" "c10mop5" "c10mop6" "c10mop7" "c10mop8" "c10mop9"
# "in1kmop1"

# "re21" "re23" "re33" "re34" "re37" "re42" "re61" "dtlz1" "dtlz2" "dtlz3" "mo_hopper_v2"
# "zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "vlmop2" "vlmop3" "omnitest" "kursawe"
# "zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "vlmop2" "vlmop3" "omnitest" "kursawe" "dtlz4" "dtlz5" "dtlz6" "dtlz7" "re21" "re23" "re33" "re34" "re37" "re42" "re61" "dtlz1" "dtlz2" "dtlz3" "mo_hopper_v2" "mo_swimmer_v2"

# MOCO Problems, and Regex RFP problem
# python config_evoxbench.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
# for env_name in "mo_cvrp"; do
#     for seed in 1; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "none"; do
#                 CUDA_VISIBLE_DEVICES=3 \
#                 python scripts/multi_obj_nn.py \
#                 --env-name ${env_name} \
#                 --seed ${seed} \
#                 --normalize-y \
#                 --filter-type best \
#                 --train-data-mode ${train_data_mode} \
#                 --train-mode ${train_mode} \
#                 --reweight-mode none \
#                 --df-name "1-final-best-hv.csv"
#             done
#         done
#     done
# done

# Discrete Problems

# for env_name in "mo_nas"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 python config_evoxbench.py

#                 CUDA_VISIBLE_DEVICES=3 \
#                 python scripts/multi_obj_nn.py \
#                 --env-name ${env_name} \
#                 --seed ${seed} \
#                 --normalize-x \
#                 --normalize-y \
#                 --discrete \
#                 --filter-type best \
#                 --train-data-mode ${train_data_mode} \
#                 --train-mode ${train_mode} \
#                 --reweight-mode none \
#                 --df-name "1-final-best-hv.csv" \
#                 --mo-solver mobo
#             done
#         done
#     done
# done
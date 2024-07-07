python config_evoxbench.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
for env_name in "re21"; do
    for seed in 1; do
        for train_data_mode in "onlybest_1"; do
            for train_mode in "grad_norm" "pcgrad"; do
                for solver in "nsga2"; do
                    for output_size in 256 32; do 
                        python config_evoxbench.py
                        CUDA_VISIBLE_DEVICES=2 \
                        python scripts/multi_head_nn.py \
                        --env-name ${env_name} \
                        --seed ${seed} \
                        --normalize-x \
                        --normalize-y \
                        --num-solutions ${output_size} \
                        --filter-type best \
                        --train-data-mode ${train_data_mode} \
                        --train-mode ${train_mode} \
                        --mo-solver ${solver} \
                        --reweight-mode none \
                        --df-name "1-test-hv.csv"
                    done
                done
            done
        done
    done
done

# for env_name in "c10mop1" "c10mop2" "c10mop3" "c10mop4" "c10mop5" "c10mop6" "c10mop7" "c10mop8" "c10mop9"; do
#     for seed in 2; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "grad_norm" "pcgrad"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "2-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "c10mop1" "c10mop2" "c10mop3" "c10mop4" "c10mop5" "c10mop6" "c10mop7" "c10mop8" "c10mop9"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "1-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "c10mop1" "c10mop2" "c10mop3" "c10mop4" "c10mop5" "c10mop6" "c10mop7" "c10mop8" "c10mop9"; do
#     for seed in 2; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "2-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "in1kmop1" "in1kmop2" "in1kmop3" "in1kmop4" "in1kmop5" "in1kmop6" "in1kmop7" "in1kmop8" "in1kmop9"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "1-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "in1kmop1" "in1kmop2" "in1kmop3" "in1kmop4" "in1kmop5" "in1kmop6" "in1kmop7" "in1kmop8" "in1kmop9"; do
#     for seed in 2; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "2-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "in1kmop1" "in1kmop2" "in1kmop3" "in1kmop4" "in1kmop5" "in1kmop6" "in1kmop7" "in1kmop8" "in1kmop9"; do
#     for seed in 1; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "grad_norm" "pcgrad"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "1-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "in1kmop1" "in1kmop2" "in1kmop3" "in1kmop4" "in1kmop5" "in1kmop6" "in1kmop7" "in1kmop8" "in1kmop9"; do
#     for seed in 2; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "grad_norm" "pcgrad"; do
#                 for solver in "nsga2"; do
#                     for output_size in 256 32; do 
#                         python config_evoxbench.py
#                         CUDA_VISIBLE_DEVICES=1 \
#                         python scripts/multi_head_nn.py \
#                         --env-name ${env_name} \
#                         --seed ${seed} \
#                         --normalize-y \
#                         --num-solutions ${output_size} \
#                         --filter-type best \
#                         --train-data-mode ${train_data_mode} \
#                         --train-mode ${train_mode} \
#                         --mo-solver ${solver} \
#                         --reweight-mode none \
#                         --df-name "2-test-hv.csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# for env_name in "motsp_100" "motsp_50" "motsp_20"; do
#     for seed in 1; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "pcgrad" "grad_norm"; do
#                 for output_size in 256; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_head_nn.py \
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
#                 for output_size in 256; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_head_nn.py \
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
#                     python scripts/multi_head_nn.py \
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
#                     python scripts/multi_head_nn.py \
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
#     for seed in 2; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "pcgrad" "grad_norm"; do
#                 for output_size in 256 32; do
#                     python config_evoxbench.py &&  \
#                     CUDA_VISIBLE_DEVICES=2 \
#                     python scripts/multi_head_nn.py \
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
#                     python scripts/multi_head_nn.py \
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

# python config_evoxbench.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
# for env_name in "regex" "rfp"; do
#     for seed in 1; do
#         for train_data_mode in "none"; do
#             for train_mode in "none"; do
#                 python config_evoxbench.py
#                 CUDA_VISIBLE_DEVICES=3 \
#                 python scripts/multi_head_nn.py \
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

# python config_evoxbench.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
# for env_name in "molecule"; do
#     CUDA_VISIBLE_DEVICES=3 \
#     python scripts/multi_head_nn.py \
#     --env-name ${env_name} \
#     --normalize-x \
#     --normalize-y \
#     --filter-type best \
#     --train-data-mode none \
#     --reweight-mode none \
#     --retrain-model 
# done

# "zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "vlmop2" "vlmop3" "omnitest" "kursawe" "mo_swimmer_v2"
# "re21" "re23" "re33" "re34" "re37" "re42" "re61" "dtlz2" "mo_hopper_v2"
# "zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "vlmop2" "vlmop3" "omnitest" "kursawe" "dtlz4" "dtlz5" "dtlz6" "dtlz7" "re21" "re23" "re33" "re34" "re37" "re42" "re61" "dtlz1" "dtlz2" "dtlz3" "mo_hopper_v2" "mo_swimmer_v2"
# python config_evoxbench.py

# for env_name in "mo_nas"; do
#     for seed in 1; do
#         for train_data_mode in "onlybest_1"; do
#             for train_mode in "grad_norm" "pcgrad"; do
#                 python config_evoxbench.py

#                 CUDA_VISIBLE_DEVICES=3 \
#                 python scripts/multi_head_nn.py \
#                 --env-name ${env_name} \
#                 --seed ${seed} \
#                 --normalize-x \
#                 --normalize-y \
#                 --discrete \
#                 --filter-type best \
#                 --train-data-mode ${train_data_mode} \
#                 --train-mode ${train_mode} \
#                 --retrain-model \
#                 --reweight-mode none \
#                 --df-name "1-final-best-hv.csv"
#             done
#         done
#     done
# done
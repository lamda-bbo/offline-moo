# For MOBO-JES, --train-gp-data-size is recommended to set as 32

# python config_evoxbench.py
# "c10mop1" "c10mop2" "c10mop3" "c10mop4" "c10mop5" "c10mop6" "c10mop7" "c10mop8" "c10mop9" "c10mop10"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
for env_name in "regex"; do
    for seed in 1; do
        for output_size in 256; do
            for train_mode in "parego"; do
                python config_evoxbench.py 
                CUDA_VISIBLE_DEVICES=1 \
                python scripts/run_mobo_once.py \
                --env-name ${env_name} \
                --sequence \
                --normalize-y \
                --num-solutions ${output_size} \
                --train-mode ${train_mode} \
                --filter-type best \
                --seed ${seed} \
                --train-gp-data-size 256 \
                --df-name $seed-test-hv.csv
            done
        done
    done
done

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
# for env_name in "mokp_100" "mokp_50"; do
#     for seed in 2; do
#         for output_size in 256; do
#             python config_evoxbench.py && \
#             CUDA_VISIBLE_DEVICES=1 \
#             python scripts/run_mobo_once.py \
#             --env-name ${env_name} \
#             --permutation \
#             --normalize-y \
#             --num-solutions ${output_size} \
#             --train-mode none \
#             --filter-type best \
#             --seed ${seed} \
#             --train-gp-data-size 256 \
#             --df-name "2-test-hv.csv"
#         done
#     done
# done

# for env_name in "zinc"; do
#     for seed in 2; do
#         for output_size in 256 32; do
#             python config_evoxbench.py && \
#             CUDA_VISIBLE_DEVICES=0 \
#             python scripts/run_mobo_once.py \
#             --env-name ${env_name} \
#             --sequence \
#             --normalize-y \
#             --num-solutions ${output_size} \
#             --train-mode none \
#             --filter-type best \
#             --seed ${seed} \
#             --train-gp-data-size 256 \
#             --df-name "2-test-hv.csv"
#         done
#     done
# done

# # python config_evoxbench.py
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
# # for env_name in "mo_swimmer_v2"; do
# #     for seed in 1; do 
# #         for size in 100; do
# #             CUDA_VISIBLE_DEVICES=0 \
# #             python scripts/run_mobo_once.py \
# #             --env-name ${env_name} \
# #             --normalize-x \
# #             --normalize-y \
# #             --seed ${seed} \
# #             --filter-type best \
# #             --train-gp-data-size ${size} \
# #             --df-name "1-final-best-hv.csv"
# #         done
# #     done
# # done

# # "zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "vlmop2" "vlmop3" "omnitest" "kursawe" "dtlz4" "dtlz5" "dtlz6" "dtlz7" "re21" "re23" "re33" "re34" "re37" "re42" "re61" "dtlz1" "dtlz2" "dtlz3" "mo_hopper_v2" "mo_swimmer_v2"
import os 
os.system(
    '''
        python config_evoxbench.py; 

        CUDA_VISIBLE_DEVICES=0 \
        python config_evoxbench.py && \
        python scripts/multi_obj_nn.py \
        --env-name molecule \
        --seed 2024 \
        --normalize-x \
        --normalize-y \
        --retrain-model \
        --num-solutions 256 \
        --filter-type best \
        --train-data-mode none \
        --train-mode none \
        --reweight-mode none \
        --df-name "test-install.csv";
        
        python config_evoxbench.py; 
        
        CUDA_VISIBLE_DEVICES=0 \
        python scripts/multi_head_nn.py \
        --env-name re21 \
        --seed 2024 \
        --normalize-x \
        --normalize-y \
        --retrain-model \
        --num-solutions 256 \
        --filter-type best \
        --train-data-mode none \
        --train-mode none \
        --reweight-mode none \
        --df-name "test-install.csv";

        python config_evoxbench.py; 

        CUDA_VISIBLE_DEVICES=0 \
        python scripts/run_mobo_once.py \
        --env-name c10mop3 \
        --seed 2024 \
        --sequence \
        --normalize-y \
        --num-solutions 32 \
        --filter-type best \
        --train-mode none \
        --train-gp-data-size 10 \
        --reweight-mode none \
        --df-name "test-install.csv"
                
    '''
)
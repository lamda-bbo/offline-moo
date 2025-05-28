import os 
import json
import sys 
from time import time 

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_PATH)

from off_moo_baselines.paretoflow.paretoflow_args import parse_args
from off_moo_baselines.paretoflow.paretoflow_experiments import (
    evaluation,
    sampling,
    train_flow_matching,
    train_proxies,
)

def run(config: dict):
    args = parse_args()
    sample_store_path = os.path.join(config['results_dir'], 'generated_samples')
    results_store_path = config['results_dir']
    config_store_path = os.path.join(config['results_dir'], 'log_configs')
    proxies_store_path = config['model_save_dir']
    fm_store_path = os.path.join(config['model_save_dir'], "flow_matching_models")
    os.makedirs(sample_store_path, exist_ok=True)
    os.makedirs(results_store_path, exist_ok=True)
    os.makedirs(config_store_path, exist_ok=True)
    os.makedirs(proxies_store_path, exist_ok=True)
    os.makedirs(fm_store_path, exist_ok=True)

    args.__dict__.update(
        {"task_name": config['task'],
         "seed": config['seed'],
         "samples_store_path": sample_store_path,
         "results_store_path": results_store_path,
         "config_store_path": config_store_path,
         "proxies_store_path": proxies_store_path
         }
    )

    name = args.task_name + "_" + str(args.seed)
    config_dict = vars(args)
    with open(os.path.join(args.config_store_path, f"{name}.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    start_time = time()
    train_proxies(args)
    train_flow_matching(args)
    sampling(args)
    print(f"Total time: {time() - start_time} seconds")


if __name__ == "__main__":
    from utils import process_args

    config = process_args(return_dict=True)

    results_dir = os.path.join(BASE_PATH, "results")
    model_save_dir = os.path.join(BASE_PATH, "model")

    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir

    run(config)
import os
import sys
import yaml
import datetime
import random
import psutil
import numpy as np

import logging
logging.root.name = 'BBOPlace-miniBench'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmarks")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
SRC_DIR = os.path.join(ROOT_DIR, "src")
UTILS_DIR = os.path.join(SRC_DIR, "bboplace_utils")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

sys.path.extend(
    [ROOT_DIR, BENCHMARK_DIR, CONFIG_DIR, SRC_DIR, UTILS_DIR]
)
os.environ["PYTHONPATH"] = ":".join(sys.path)

def parse_args(args):
    setattr(args, "ROOT_DIR", ROOT_DIR)
    setattr(args, "BENCHMARK_DIR", BENCHMARK_DIR)
    setattr(args, "CONFIG_DIR", CONFIG_DIR)
    setattr(args, "SRC_DIR", SRC_DIR)
    setattr(args, "UTILS_DIR", UTILS_DIR)
    setattr(args, "RESULTS_DIR", RESULTS_DIR)

    
    # default config
    with open(os.path.join(CONFIG_DIR, "default.yaml"), 'r') as f:
        default_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = update_args(args, default_config_dict)

    # benchmark config
    benchmark_base, design_name = args.benchmark.split('/')
    setattr(args, "benchmark_base", benchmark_base)
    setattr(args, "design_name", design_name)
    setattr(args, "benchmark_path", os.path.join(BENCHMARK_DIR, benchmark_base, design_name))
    with open(os.path.join(CONFIG_DIR, "benchmarks", f"{benchmark_base}.yaml"), 'r') as f:
        benchmark_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = update_args(args, benchmark_config_dict)

    # placer config
    placer_config_path = os.path.join(CONFIG_DIR, "placer", f"{args.placer}.yaml")
    if os.path.exists(placer_config_path):
        with open(placer_config_path, 'r') as f:
            placer_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        args = update_args(args, placer_config_dict)

    # algorithm config
    algorithm_config_path = os.path.join(CONFIG_DIR, "algorithm", f"{args.algorithm}.yaml")
    if os.path.exists(algorithm_config_path):
        with open(os.path.join(CONFIG_DIR, "algorithm", f"{args.algorithm}.yaml"), 'r') as f:
            algorithm_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        args = update_args(args, algorithm_config_dict)

    # set seed
    set_seed(args.seed)

    # set unique token and result path
    unique_token = "seed_{}_{}".format(args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    setattr(args, "unique_token", unique_token)
    setattr(
        args, 
        "result_path", 
        os.path.join(
            RESULTS_DIR, 
            "{}/{}/{}/{}/{}".format(
                args.design_name,
                args.name,
                args.placer,
                args.algorithm,
                args.unique_token
            )
        )
    )
    os.makedirs(args.result_path, exist_ok=True)

    cpus = psutil.cpu_count(logical=True)
    setattr(args, "n_cpu", min(args.n_cpu, cpus))

    return args

def update_args(args, config_dict:dict):
    for key, value in config_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
# BBOPlace-miniBench: Mini Benchmarking Black-Box Optimization for Chip Placement

This repository contains the Python code for BBOPlace-miniBench, a mini benchmarking of BBOPlace-Bench without DREAMPlace . 

## Requirements
+ numpy==1.24.4
+ ray==2.10.0
+ matplotlib==3.7.5
+ igraph==0.11.8
+ pymoo==0.6.1.2
+ torch==2.4.1
+ torchaudio==2.4.1
+ torchvision==0.19.1
+ gpytorch==1.10
+ botorch==0.8.5
+ pypop7==0.0.82

## File structure

+ `benchmarks` directory stores the benchmarks for running. Please download ISPD2005 and ICCAD2015 benchmarks and move them to `benchmarks/` (i.e., `benchmarks/ispd2005/adaptec1`, `benchmark/iccad2015/superblue1`).
+ `config` stores the hyperparameters for algorithms.
+ `script` contains scripts for code running.
+ `src` contains the source code of our benchmarking.
  
## Usage
Please first build the environment according to the requirements and download benchmarks via google drive: [ISPD2005](https://drive.google.com/drive/folders/1MVIOZp2rihzIFK3C_4RqJs-bUv1TW2YT?usp=sharing), [ICCAD2015](https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing).


### Usage Demo and Evaluation APIs (for BBO Users)
We provide a simple demo of how to benchmark BBO algorithms (e.g., CMA-ES) on our proposed BBOPlace-Bench in `src/demo_cmaes.py`. 
Specifically, we implement a BBO user-friendly evaluator interface in `src/evaluator.py`, and you can instantiate it for fitness evaluation as
```python
# Set base_path environment variable
import os, sys
base_path = os.path.abspath(".") # Alternative: base_path should be where BBOPlace-miniBench is located
sys.path.append(base_path)

import numpy as np 
from types import SimpleNamespace
from src.evaluator import Evaluator
from utils.args_parser import parse_args

args = SimpleNamespace(
    **{
        "placer": "gg", # GG in our paper
        "benchmark": "ispd2005/adaptec1", # choose which placement benchmark
    } 
)

# Read config (i.e. benchmark, placer)
args = parse_args(args)

# Instantiate the evaluator
evaluator = Evaluator(args)

# Read problem metadata
dim: int = evaluator.n_dim
xl: np.ndarray = evaluator.xl
xu: np.ndarray = evaluator.xu
assert len(xl) == len(xu) == dim

# Hpwl evaluation API
batch_size = 16
x = np.random.uniform(low=xl, high=xu, size=(batch_size, dim))
hpwl, macor_pos = evaluator.evaluate(x)
print(np.max(hpwl), np.min(hpwl), np.mean(hpwl))
```
where choices for `placer` are `gg`, `sp`, which refer to GG, SP formulations in our paper, respectively. The choices for `benchmark` are 
```python
benchmarks = ["ispd2005/adaptec1", "ispd2005/adaptec2", "ispd2005/adaptec3", "ispd2005/adaptec4", "ispd2005/bigblue1", "ispd2005/bigblue3",   # ISPD 2005
              "iccad2015/superblue1", "iccad2015/superblue3", "iccad2015/superblue4", "iccad2015/superblue5",
              "iccad2015/superblue7", "iccad2015/superblue10", "iccad2015/superblue16", "iccad2015/superblue18"]   # ICCAD 2015
```

### Search Space Statement

For GG formulation, we formulate it as continuous BBO problems. For SP formulation, it is a permutation-based BBO problem.

## Reproduce Paper Results

### Parameters Settings
Before running an experiment, you can modify the hyper-parameters of different problem formulations and algorithms in `config` directory. For example, modifying `n_population : 50` in `config/algorithm/ea.yaml` to change the population size of EA.

### Quick Run

Run ``run.sh`` file in the `script` directory.
```shell
cd script
```
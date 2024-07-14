import ray 
import os 
import psutil
import torch 
import datetime 
from typing import List

BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".."
)

def SyntheticFunction(train_mode: str = "Vallina",
                      tasks: List[str] = [],
                      cpus = psutil.cpu_count(logical=True),
                      gpus = torch.cuda.device_count(),
                      num_parallel = 1,
                      num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import SyntheticFunction as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def RESuite(train_mode: str = "Vallina",
                tasks: List[str] = [],
                cpus = psutil.cpu_count(logical=True),
                gpus = torch.cuda.device_count(),
                num_parallel = 1,
                num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import RESuite as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def MONASSequence(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import MONASSequence as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': False,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def MONASLogits(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import MONASLogits as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': True,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def MOCOPermutation(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import MOCOPermutation as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': False,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def MOCOContinuous(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import MOCOContinuous as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def MORL(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import MORL as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def ScientificDesignContinuous(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import ScientificDesignContinuous as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': True,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
    
def ScientificDesignSequence(train_mode: str = "Vallina",
            tasks: List[str] = [],
            cpus = psutil.cpu_count(logical=True),
            gpus = torch.cuda.device_count(),
            num_parallel = 1,
            num_samples = 1):
    
    from off_moo_baselines.multi_head import multi_head_run 
    from off_moo_bench.task_set import ScientificDesignSequence as tasks_to_run
    from ray.tune import run
    
    if len(tasks) == 0:
        tasks = tasks_to_run
    else:
        for task in tasks:
            assert task in tasks_to_run
            
    
    name = f"MultiHead-{train_mode}"
    
    results_dir = os.path.join(BASE_PATH, "results")
    ray_dir = os.path.join(BASE_PATH, "ray_results", name)
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(ray_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        _temp_dir=os.path.expanduser('~/tmp'),
    )
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-{ts.year}-{ts.month:02d}-{ts.day:02d}-{ts.hour:02d}-{ts.minute:02d}-{ts.second:02d}'
    
    seeds = [1000, 2000, 3000, 4000, 5000]
    
    run(
        multi_head_run,
        name=name + ts_name,
        config={
            'results_dir': results_dir,
            'model_save_dir': model_dir,
            'use_wandb': False,
            'wandb_api': '9f59486bed008c431a4a5804c35bb3c065d0b658',
            'run_type': 'debug',
            'seed': ray.tune.grid_search(seeds),
            'model': 'MultiHead',
            'train_mode': train_mode,
            'retrain_model': False,
            'data_pruning': False,
            'data_preserved_ratio': 0.2, 
            'task': ray.tune.grid_search(tasks),
            'normalize_xs': False,
            'normalize_ys': True,
            'to_logits': False,
            'n_epochs': 200,
            'batch_size': 32,
            'forward_lr': 1e-3,
            'forward_lr_decay': 0.98,
            'solver_n_gen': 50,
            'solver_init_method': 'nds',
            'num_solutions': 256
        },
        num_samples=num_samples,
        storage_path=results_dir,
        resources_per_trial={'cpu': cpus // num_parallel,
                             'gpu': gpus / num_parallel - 0.01 if gpus / num_parallel < 1 \
                                 else gpus // num_parallel}
    )
import off_moo_bench as ob 
import os 
import wandb 
import torch 
import numpy as np 
import datetime 
import json 
import matplotlib.pyplot as plt 
from copy import deepcopy
from pymoo.algorithms.moo.nsga2 import NSGA2
from utils import set_seed, get_quantile_solutions
from off_moo_baselines.multiple_models.nets import MultipleModels
from off_moo_baselines.multiple_models.trainer import get_trainer
from off_moo_baselines.multiple_models.surrogate_problem import MultipleSurrogateProblem
from off_moo_baselines.mo_solver.moea_solver import MOEASolver
from off_moo_baselines.mo_solver.callback import RecordCallback
from off_moo_baselines.data import tkwargs, get_dataloader
from off_moo_bench.evaluation.metrics import hv
from off_moo_bench.evaluation.plot import plot_y

def multiple_run(config):
    
    results_dir = os.path.join(config['results_dir'], 
                               f"{config['model']}-{config['train_mode']}-{config['task']}")
    config["results_dir"] = results_dir 
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"
    
    logging_dir = os.path.join(config['results_dir'], run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)

    if config['use_wandb']:
        if 'wandb_api' in config.keys():
            wandb.login(key=config['wandb_api'])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config['run_type'],
            mode="online",
            dir=config['results_dir']
        )
    
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config['seed'])

    task = ob.make(config['task'])
    
    X = task.x.copy()
    y = task.y.copy()
    
    if config["data_pruning"]:
        X, y = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config["data_preserved_ratio"]),
            return_x=True, return_y=True
        )
    
    X_test = task.x_test.copy()
    y_test = task.y_test.copy()
    
    if config['normalize_xs']:
        task.map_normalize_x()
        X = task.normalize_x(X)
        X_test = task.normalize_x(X_test)
    if config['to_logits']:
        assert task.is_discrete 
        task.map_to_logits()
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    if config['normalize_ys']:
        task.map_normalize_y()
        y = task.normalize_y(y)
        y_test = task.normalize_y(y_test)
    
    data_size, n_dim = tuple(X.shape)
    n_obj = y.shape[1]
        
    model_save_dir = config['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)
    
    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode=config['train_mode'],
        hidden_size=[2048, 2048],
        save_dir=config['model_save_dir'],
        save_prefix=f"{config['model']}-{config['train_mode']}-{config['task']}-{config['seed']}"
    )
    assert 0, model.obj2model[0].models
    model.set_kwargs(**tkwargs)
    
    trainer_func = get_trainer(config["train_mode"])
    
    for which_obj in range(n_obj):
        
        y0 = y[:, which_obj].copy().reshape(-1, 1)
        y0_test = y_test[:, which_obj].copy().reshape(-1, 1)
        
        config["which_obj"] = which_obj
        config["input_shape"] = X[0].shape
        
        indexs = np.argsort(y0.squeeze())
        index = indexs[:config["num_solutions"]]
        config["best_x"] = torch.from_numpy(deepcopy(X[index])).to(**tkwargs)
        config["best_y"] = torch.from_numpy(deepcopy(y[index])).to(**tkwargs)
    
        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj], 
            config=config,
        )
        
        (
            train_loader,
            val_loader,
            test_loader
        ) = get_dataloader(X, y0, X_test, y0_test,
                        val_ratio=0.9,
                        batch_size=config["batch_size"])
        
        trainer.launch(
            train_loader,
            val_loader,
            test_loader,
            retrain_model=config["retrain_model"]
        )
    
    # if config['use_wandb']:
    #     wandb.init(
    #         project="Offline-MOO",
    #         name=run_name + ts_name,
    #         config=config,
    #         group=f"End2End-{config['train_mode']}",
    #         job_type="search",
    #         mode="online",
    #         dir=logging_dir,
    #         reinit=True
    #     )
    
    surrogate_problem = MultipleSurrogateProblem(
        n_var=n_dim, n_obj=n_obj, model=model
    )
    
    callback = RecordCallback(
        task=task, surrogate_problem=surrogate_problem,
        config=config, logging_dir=logging_dir, iters_to_record=1
    )
    
    solver = MOEASolver(
        n_gen=config["solver_n_gen"],
        pop_init_method=config["solver_init_method"],
        batch_size=config["num_solutions"],
        pop_size=config["num_solutions"],
        algo=NSGA2,
        callback=callback,
        eliminate_duplicates=True,
    )
    
    res = solver.solve(surrogate_problem, X=X, Y=y)
    
    res_x = res["x"]
    if config['normalize_xs']:
        task.map_denormalize_x()
        res_x = task.denormalize_x(res_x)
    
    res_y = task.predict(res_x)
    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]
    
    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)
    
    nadir_point = task.nadir_point
    if config['normalize_ys']:
        res_y = task.normalize_y(res_y)
        nadir_point = task.normalize_y(nadir_point)
        res_y_50_percent = task.normalize_y(res_y_50_percent)
        res_y_75_percent = task.normalize_y(res_y_75_percent)
        
    _, d_best = task.get_N_non_dominated_solutions(
        N=config["num_solutions"], 
        return_x=False, return_y=True
    )
    
    plot_y(res_y, save_dir=logging_dir, config=config,
           nadir_point=nadir_point, d_best=d_best)
        
    d_best_hv = hv(nadir_point, d_best, config['task'])
    hv_value = hv(nadir_point, res_y, config['task'])
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, config['task'])
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, config['task'])
    
    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")
    
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value, 
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
        "evaluation_step": 1,
    }
    
    if config["use_wandb"]:
        wandb.log(hv_results)


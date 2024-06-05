import off_moo_bench as ob 
import os 
import wandb 
import numpy as np 
import datetime 
import json 

def run(config):
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"End2End-{config['train_mode']}-seed{config['seed']}"
    
    logging_dir = config['logging_dir']

    wandb.init(
        project="Offline MOO",
        name=run_name + ts_name,
        config=config,
        dir=logging_dir
    )
    
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    task = ob.make(config['task'])
    
    x = task.x 
    y = task.y 
    
    x_test = task.x_test.copy()
    y_test = task.y_test.copy()
    
    if config['normalize_xs']:
        task.map_normalize_x()
        x_test = task.normalize_x(x_test)
    if task.is_discrete:
        task.map_to_logits()
        x_test = task.to_logits(x_test)
    if config['normalize_ys']:
        task.map_normalize_y()
        y_test = task.normalize_y(y_test)
    
    
    
    
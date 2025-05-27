import ray
import numpy as np
import logging
import os

from utils.args_parser import parse_args
from placedb import PlaceDB
from placer import REGISTRY as PLACER_REGISTRY


@ray.remote(num_cpus=1)  
def evaluate_placer(placer, x0):
    return placer.evaluate(x0)

class Evaluator:
    def __init__(self, args):
        args = parse_args(args)
        self.args = args
        print(f"Number of CPUs configured: {args.n_cpu}")

        self.placedb = PlaceDB(args=args)
        self.placer = PLACER_REGISTRY[args.placer.lower()](args=args, placedb=self.placedb)

        if ray.is_initialized():
            ray.shutdown()
            
        ray.init(
            num_cpus=args.n_cpu,
            num_gpus=1,
            include_dashboard=False,
            logging_level=logging.INFO,
            _temp_dir=os.path.expanduser('~/tmp'),
            log_to_driver=True
        )
        
        print("Ray resources:", ray.cluster_resources())
        logging.info("Finished initializing evaluator")

    @property
    def node_cnt(self):
        return self.placedb.node_cnt 

    @property
    def n_dim(self):
        node_cnt = self.placedb.node_cnt 
        return node_cnt * 2
    
    @property
    def xl(self):
        node_cnt = self.placedb.node_cnt 
        return np.zeros(node_cnt * 2)

    @property
    def xu(self):
        if self.args.placer == "gg":
            node_cnt = self.placer.placedb.node_cnt 
            n_grid_x = self.args.n_grid_x
            n_grid_y = self.args.n_grid_y 
            return np.array(
                ([n_grid_x] * node_cnt) + ([n_grid_y] * node_cnt)
            )
        elif self.args.placer == "sp":
            node_cnt = self.placedb.node_cnt 
            return np.array([node_cnt] * node_cnt * 2)
        else:
            raise ValueError(f"Not supported placer {self.args.placer}") 

    def evaluate(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if x.shape == 1:
            x = x.reshape(1, -1)

        if self.args.n_cpu > 1 and x.shape[0] > 1:
            futures = [evaluate_placer.remote(self.placer, x0) for x0 in x]
            results = ray.get(futures)
        else:
            results = [self.placer.evaluate(x0) for x0 in x]
        hpwl_lst       = [result[0]["hpwl"] for result in results]
        congestion_lst = [result[0]["congestion"] for result in results]
        regularity_lst = [result[0]["regularity"] for result in results]
        macro_pos_lst  = [result[1] for result in results]

        res = {
            "hpwl"       : np.array(hpwl_lst),
            "congestion" : np.array(congestion_lst),
            "regularity" : np.array(regularity_lst),
        }
        return res, macro_pos_lst 
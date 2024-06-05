import os 
import pickle 

def class2dict(data):
    dicdadta = {}
    for name in dir(data):
        value = getattr(data, name)
        if (not name.startswith('__')) and (not callable(value)):
            dicdadta[name] = value
    return dicdadta

with open("m2bo_bench/problem/lambo/data/experiments/test/proxy_rfp_problem.pkl", 'rb+') as f:
    print(class2dict(pickle.load(f)))
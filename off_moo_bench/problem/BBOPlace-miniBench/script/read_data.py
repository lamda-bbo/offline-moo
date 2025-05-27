import numpy as np 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='adaptec1')
args = parser.parse_args() 
base_dir = f"./nsgaii_results_0526_1816/ispd2005/{args.task}"
x = np.load(f'{base_dir}/all_X.npy')
f = np.load(f'{base_dir}/all_F.npy')
print(x.shape, f.shape)
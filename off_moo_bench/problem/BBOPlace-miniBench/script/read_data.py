import numpy as np 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='adaptec1')
args = parser.parse_args() 
base_dir = f"./nsgaii_results_0526_1816/ispd2005/{args.task}"
x = np.load(f'{base_dir}/all_X.npy')
f = np.load(f'{base_dir}/all_F.npy')
print(x.shape, f.shape)

import matplotlib.pyplot as plt

# 创建散点图
plt.figure(figsize=(10, 8))
plt.scatter(f[:, 0], f[:, 1], alpha=0.6)

# 添加标签和标题
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title(f'Objective Space Visualization for {args.task}')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
plt.savefig('test.png')
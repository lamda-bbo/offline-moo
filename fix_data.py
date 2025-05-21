import numpy as np

task_names = ["dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6"]

for task_name in task_names:
    data = np.load(f"./data/{task_name}/{task_name}-y-0.npy")
    print(data.shape)

import json
import os

import numpy as np

import off_moo_bench as ob
from off_moo_bench.task_set import ALLTASKSDICT

problematic_tasks = []
task_dict = {
    "rfp": "RFP-Exact-v0",
    #  "dtlz3": "DTLZ3-Exact-v0",
    #  "dtlz4": "DTLZ4-Exact-v0",
    #  "dtlz6": "DTLZ6-Exact-v0",
}

for small_name, full_name in task_dict.items():
    try:
        task = ob.make(full_name)
        y_true = task.y[:50]
        y_pred = task.predict(task.x[:50])

        print(y_true.shape, y_pred.shape)
        print(y_true)
        print(np.abs(y_true - y_pred))

        max_diff = np.max(np.abs(y_true - y_pred))
        if max_diff > 1e-3:  
            problematic_tasks.append(f"{small_name}: big predictions differences ({max_diff:.6f})")
            continue

        # save
        fronts = [front.tolist() for front in task.dataset.fronts]
        os.makedirs(f"./data/{small_name}", exist_ok=True)
        with open(
            f"./data/{small_name}/{small_name}_fronts.json", "w", encoding="utf-8"
        ) as f:
            json.dump(fronts, f, indent=4)

    except Exception as e:
        problematic_tasks.append(f"{small_name}: {str(e)}")

if problematic_tasks:
    with open("problematic_tasks.txt", "w", encoding='utf-8') as f:
        for task in problematic_tasks:
            f.write(f"{task}\n")

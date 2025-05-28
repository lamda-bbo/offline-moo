import json
import os

import numpy as np
import ray

import off_moo_bench as ob
from off_moo_bench.task_set import ALLTASKSDICT


@ray.remote
def process_task(small_name, full_name):
    # try:
    task = ob.make(full_name)
    y_true = task.y[:10]
    y_pred = task.predict(task.x[:10])

    print(y_true.shape, y_pred.shape)
    print(y_true)
    print(np.abs(y_true - y_pred))

    max_diff = np.max(np.abs(y_true - y_pred))
    if max_diff > 1e-3:
        return f"{small_name}: big predictions differences ({max_diff:.6f})"

    # save
    fronts = [front.tolist() for front in task.dataset.fronts]
    os.makedirs(f"./data/{small_name}", exist_ok=True)
    with open(
        f"./data/{small_name}/{small_name}_fronts.json", "w", encoding="utf-8"
    ) as f:
        json.dump(fronts, f, indent=4)

    return None


# except Exception as e:
#     return f"{small_name}: {str(e)}"


task_dict = {
    # "c10mop1": "C10MOP1-Exact-v0",
    # "c10mop2": "C10MOP2-Exact-v0",
    # "c10mop5": "C10MOP5-Exact-v0",
    # "c10mop6": "C10MOP6-Exact-v0",
    # "c10mop7": "C10MOP7-Exact-v0",
    # "c10mop8": "C10MOP8-Exact-v0",
    # "c10mop9": "C10MOP9-Exact-v0",
    # "in1kmop1": "IN1KMOP1-Exact-v0",
    # "in1kmop2": "IN1KMOP2-Exact-v0",
    # "in1kmop3": "IN1KMOP3-Exact-v0",
    # "in1kmop7": "IN1KMOP7-Exact-v0",
    # "in1kmop8": "IN1KMOP8-Exact-v0",
    # "in1kmop9": "IN1KMOP9-Exact-v0",
    # "nb201_test": "NASBench201Test-Exact-v0",
    # "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    # "mo_hopper_v2": "MOHopperV2-Exact-v0",
    "rfp": "RFP-Exact-v0",
    "dtlz2": "DTLZ2-Exact-v0",
    "dtlz3": "DTLZ3-Exact-v0",
    "dtlz4": "DTLZ4-Exact-v0",
    "dtlz5": "DTLZ5-Exact-v0",
    "dtlz6": "DTLZ6-Exact-v0",
    "dtlz7": "DTLZ7-Exact-v0",
}

ray.init()

futures = [
    process_task.remote(small_name, full_name)
    for small_name, full_name in task_dict.items()
]

results = ray.get(futures)

problematic_tasks = [result for result in results if result is not None]

ray.shutdown()

if problematic_tasks:
    with open("problematic_tasks.txt", "w", encoding="utf-8") as f:
        for task in problematic_tasks:
            f.write(f"{task}\n")

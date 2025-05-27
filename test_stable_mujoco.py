import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import ray

import off_moo_bench as ob


@ray.remote
def run_single_trial(task, x):
    return task.predict(x)


def analyze_single_task(
    small_name: str,
    full_name: str,
    n_trials: int = 5,
    error_threshold: float = 1e-3,
    variance_threshold: float = 1e-5,
) -> Dict:
    print(f"\nAnalyzing task: {small_name}")

    task = ob.make(full_name)
    n_objectives = task.y.shape[1]

    print(f"Number of objectives: {n_objectives}")

    # Parallelize trials using Ray
    ray.init(ignore_reinit_error=True)
    futures = [run_single_trial.remote(task, task.x[:]) for _ in range(n_trials)]
    full_predictions = np.array(ray.get(futures))
    ray.shutdown()

    mean_prediction = np.mean(full_predictions, axis=0)

    # at each objectives
    objective_analysis = {}
    update_needed = False
    unstable_objectives = []
    significant_diff_objectives = []

    for obj_idx in range(n_objectives):
        obj_predictions = full_predictions[:, :, obj_idx]
        obj_variance = np.mean(np.var(obj_predictions, axis=0))

        obj_diff = np.abs(task.y[:, obj_idx] - mean_prediction[:, obj_idx])

        # print(f'obj{obj_idx}', obj_diff)
        # continue

        obj_stats = {
            "prediction_variance": float(obj_variance),
            "max_difference": float(np.max(obj_diff)),
            "mean_difference": float(np.mean(obj_diff)),
            "std_difference": float(np.std(obj_diff)),
            "is_stable": obj_variance < variance_threshold,
            "has_significant_diff": np.max(obj_diff) > error_threshold,
        }

        objective_analysis[f"objective_{obj_idx}"] = obj_stats

        if not obj_stats["is_stable"]:
            unstable_objectives.append(obj_idx)
        if obj_stats["has_significant_diff"]:
            significant_diff_objectives.append(obj_idx)

        if obj_stats["has_significant_diff"]:  # and obj_stats["is_stable"]:
            update_needed = True

    # assert 0

    update_recommendation = {
        "should_update": update_needed,
        "reason": "",
        "unstable_objectives": unstable_objectives,
        "significant_diff_objectives": significant_diff_objectives,
    }

    result = {
        "n_objectives": n_objectives,
        "objective_analysis": objective_analysis,
        "update_recommendation": update_recommendation,
    }

    if update_needed:
        save_dir = f"./data/{small_name}"
        os.makedirs(save_dir, exist_ok=True)

        np.save(f"{save_dir}/new_ground_truth.npy", mean_prediction)
        print(mean_prediction.shape)

        print(f"\nTask {small_name} requires ground truth update:")
        print(f"New ground truth saved to: {save_dir}/new_ground_truth.npy")

    return result


task_dict = {
    # "c10mop1": "C10MOP1-Exact-v0",
    # "c10mop2": "C10MOP2-Exact-v0",
    # "c10mop5": "C10MOP5-Exact-v0",
    # "c10mop6": "C10MOP6-Exact-v0",
    # "c10mop7": "C10MOP7-Exact-v0",
    # "c10mop8": "C10MOP8-Exact-v0",
    # "c10mop9": "C10MOP9-Exact-v0",
    # "in1kmop1": "IN1KMOP1-Exact-v0",
    # "in1kmop2": "IN1KMOP1-Exact-v0",
    # "in1kmop3": "IN1KMOP3-Exact-v0",
    # "in1kmop7": "IN1KMOP7-Exact-v0",
    # "in1kmop8": "IN1KMOP8-Exact-v0",
    # "in1kmop9": "IN1KMOP9-Exact-v0",
    # "nb201_test": "NASBench201Test-Exact-v0",
    # "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    # "mo_hopper_v2": "MOHopperV2-Exact-v0",
    "rfp": "RFP-Exact-v0"
}

start_time = time.time()
results = {}

for small_name, full_name in task_dict.items():
    results[small_name] = analyze_single_task(small_name, full_name)

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

for task_name, analysis in results.items():
    print(f"\nAnalysis Results for {task_name}:")
    if "error" in analysis:
        print(f"Analysis failed with error: {analysis['error']}")
        continue

    print(f"\nNumber of objectives: {analysis['n_objectives']}")

    print("\nObjective-wise Analysis:")
    for obj_name, obj_stats in analysis["objective_analysis"].items():
        print(f"\n{obj_name}:")
        print(f"Prediction variance: {obj_stats['prediction_variance']:.6f}")
        print(f"Maximum difference: {obj_stats['max_difference']:.6f}")
        print(f"Mean difference: {obj_stats['mean_difference']:.6f}")
        print(f"Is stable: {obj_stats['is_stable']}")
        print(f"Has significant difference: {obj_stats['has_significant_diff']}")

    print("\nUpdate Recommendation:")
    print(f"Should update: {analysis['update_recommendation']['should_update']}")
    print(f"Reason: {analysis['update_recommendation']['reason']}")
    if analysis["update_recommendation"]["unstable_objectives"]:
        print(
            f"Unstable objectives: {analysis['update_recommendation']['unstable_objectives']}"
        )
    if analysis["update_recommendation"]["significant_diff_objectives"]:
        print(
            f"Objectives with significant differences: {analysis['update_recommendation']['significant_diff_objectives']}"
        )

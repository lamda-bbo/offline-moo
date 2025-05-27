import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import ray

import off_moo_bench as ob


@ray.remote
def predict_batch(task, x_batch):
    return task.predict(x_batch)


def analyze_single_task(
    small_name: str,
    full_name: str,
    batch_size: int = 10,  
    error_threshold: float = 1e-3,
) -> Dict:
    print(f"\nAnalyzing task: {small_name}")

    task = ob.make(full_name)
    n_objectives = task.y.shape[1]
    total_samples = len(task.x)

    print(f"Number of objectives: {n_objectives}")
    print(f"Total samples: {total_samples}")

    ray.init(ignore_reinit_error=True)
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    predictions = []
    failed_batches = []
    valid_indices = []  
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        x_batch = task.x[start_idx:end_idx]
        
        try:
            futures = predict_batch.remote(task, x_batch)
            batch_predictions = ray.get(futures)
            predictions.append(batch_predictions)
            valid_indices.extend(range(start_idx, end_idx))
        except Exception as e:
            print(f"Error in batch {i} (indices {start_idx}-{end_idx}): {str(e)}")
            failed_batches.append({
                "batch_index": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "error": str(e)
            })
            batch_predictions = np.full((end_idx - start_idx, n_objectives), np.nan)
            predictions.append(batch_predictions)
    
    ray.shutdown()

    full_predictions = np.concatenate(predictions, axis=0)
    valid_indices = np.array(valid_indices)  

    objective_analysis = {}
    update_needed = False
    significant_diff_objectives = []

    for obj_idx in range(n_objectives):
        obj_predictions = full_predictions[:, obj_idx]

        valid_mask = ~np.isnan(obj_predictions)
        obj_diff = np.abs(task.y[valid_mask, obj_idx] - obj_predictions[valid_mask])

        obj_stats = {
            "max_difference": float(np.max(obj_diff)) if len(obj_diff) > 0 else None,
            "mean_difference": float(np.mean(obj_diff)) if len(obj_diff) > 0 else None,
            "std_difference": float(np.std(obj_diff)) if len(obj_diff) > 0 else None,
            "has_significant_diff": np.max(obj_diff) > error_threshold if len(obj_diff) > 0 else None,
            "failed_samples_count": np.sum(~valid_mask)
        }

        objective_analysis[f"objective_{obj_idx}"] = obj_stats

        if obj_stats["has_significant_diff"]:
            significant_diff_objectives.append(obj_idx)
            update_needed = True

    update_recommendation = {
        "should_update": update_needed,
        "significant_diff_objectives": significant_diff_objectives,
    }

    result = {
        "n_objectives": n_objectives,
        "objective_analysis": objective_analysis,
        "update_recommendation": update_recommendation,
        "failed_batches": failed_batches,
        "valid_samples_count": len(valid_indices)
    }

    if update_needed:
        save_dir = f"./data/{small_name}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/new_ground_truth.npy", full_predictions)
        # 保存有效的x值
        valid_x = task.x[valid_indices]
        np.save(f"{save_dir}/new_valid_x.npy", valid_x)
        print(f"\nTask {small_name} requires ground truth update:")
        print(f"New ground truth saved to: {save_dir}/new_ground_truth.npy")
        print(f"Valid x values saved to: {save_dir}/new_valid_x.npy")
        print(f"Number of valid samples: {len(valid_indices)}")

    return result


task_dict = {
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
    print(f"Total valid samples: {analysis['valid_samples_count']}")

    print("\nObjective-wise Analysis:")
    for obj_name, obj_stats in analysis["objective_analysis"].items():
        print(f"\n{obj_name}:")
        print(f"Maximum difference: {obj_stats['max_difference']:.6f}" if obj_stats['max_difference'] is not None else "Maximum difference: N/A")
        print(f"Mean difference: {obj_stats['mean_difference']:.6f}" if obj_stats['mean_difference'] is not None else "Mean difference: N/A")
        print(f"Has significant difference: {obj_stats['has_significant_diff']}")
        print(f"Failed samples count: {obj_stats['failed_samples_count']}")

    print("\nUpdate Recommendation:")
    print(f"Should update: {analysis['update_recommendation']['should_update']}")
    if analysis["update_recommendation"]["significant_diff_objectives"]:
        print(
            f"Objectives with significant differences: {analysis['update_recommendation']['significant_diff_objectives']}"
        )
    
    if analysis["failed_batches"]:
        print("\nFailed Batches:")
        for batch in analysis["failed_batches"]:
            print(f"Batch {batch['batch_index']}: indices {batch['start_idx']}-{batch['end_idx']}")
            print(f"Error: {batch['error']}")
import numpy as np

import off_moo_bench as ob

task = ob.make("Bigblue3-Exact-v0")
print(task.predict(task.x[:5]))
print(task.y[:5])
y_all = np.vstack([task.y, task.y_test])
print(",")
print(f"ideal_point={y_all.min(axis=0).tolist()},")
print(f"nadir_point={y_all.max(axis=0).tolist()},")
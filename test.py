# import os

# # 当前工作目录
# current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# # 检查目录是否存在，并进行重命名
# old_dir_name = 'mocvrp_100'
# new_dir_name = 'bi_cvrp_100'
# if os.path.exists(os.path.join(current_dir, old_dir_name)):
#     os.rename(os.path.join(current_dir, old_dir_name), 
#               os.path.join(current_dir, new_dir_name))
#     print(f"Directory renamed from {old_dir_name} to {new_dir_name}")
# else:
#     print(f"Directory '{old_dir_name}' not found.")

# # 遍历新目录中的所有文件
# new_dir_path = os.path.join(current_dir, new_dir_name)
# if os.path.exists(new_dir_path):
#     for filename in os.listdir(new_dir_path):
#         if old_dir_name in filename:
#             # 替换文件名中的字符串
#             new_filename = filename.replace(old_dir_name, new_dir_name)
#             os.rename(os.path.join(new_dir_path, filename),
#                       os.path.join(new_dir_path, new_filename))
#             print(f"Renamed '{filename}' to '{new_filename}'")
# else:
#     print(f"Directory '{new_dir_name}' does not exist or was not created.")
from evoxbench.database.init import config
import os
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'off_moo_bench', 'problem', 'mo_nas')
config(os.path.join(base_path, 'database'), os.path.join(base_path, 'data'))

print(base_path)

import off_moo_bench
from off_moo_bench import make 

task = make("ZINC-Exact-v0")

print(task.x[0])
x = task.x[:100]
print(task.x.shape, task.y.shape)
# y0 = task.predict(x)
print(task.problem.xl, task.problem.xu)
# for y, y_pred in zip(task.y[:100], y0):
#     print(y, y_pred)
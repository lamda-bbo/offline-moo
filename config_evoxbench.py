from evoxbench.database.init import config
import os
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'm2bo_bench', 'problem', 'mo_nas')
config(os.path.join(base_path, 'database'), os.path.join(base_path, 'data'))
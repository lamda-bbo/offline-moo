from off_moo_baselines.multiple.experiment_roma import SyntheticFunction
import os, sys
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

############################# config evoxbench #############################
from evoxbench.database.init import config
nas_path = os.path.join(
    base_path,
    'off_moo_bench', 'problem', 'mo_nas'
)
config(os.path.join(nas_path, 'database'), os.path.join(nas_path, 'data'))
#############################################################################

SyntheticFunction(
    tasks=["ZDT1-Exact-v0"],
    num_parallel=2,
)
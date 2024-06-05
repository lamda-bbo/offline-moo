from typing import Any
# from .psl_mobo.run import run as psl_mobo_run
# from .nsga2.run import run as nsga2_run
# from .nsga2.single_obj_run import run as single_obj_nsga2_run

# class RunAlgorithm:

#     def __init__(self, env_name):
        
#         algo_dict = {
#             "psl_mobo": {
#                 "default": psl_mobo_run
#             },
#             "nsga2": {
#                 "default": nsga2_run,
#                 "single_obj": single_obj_nsga2_run
#             }
#         }
#         if env_name not in algo_dict.keys():
#             raise Exception(f"Algorithm {env_name} not found!")
#         self.func = algo_dict[env_name]
    
#     def __call__(self, args, *ags, **kwargs):
#         return self.func[args.model_type](args, *ags, **kwargs)
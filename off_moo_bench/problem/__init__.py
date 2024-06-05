# from m2bo_bench.problem.synthetic_func import *

# load_nas = True
# try:
#     from m2bo_bench.problem.mo_nas import *
# except:
#     load_nas = False

# from m2bo_bench.problem.mo_nas import *
# # from m2bo_bench.problem.qm9 import *
# from m2bo_bench.problem.comb_opt import *
# from m2bo_bench.problem.dtlz import *

# load_morl = True
# try:
#     from m2bo_bench.problem.morl import *
# except:
#     load_morl = False
    
# from m2bo_bench.problem.morl import *

# load_lambo = True
# # print('1')
# # try:
# #     from m2bo_bench.problem.lambo import *
# # except:
# #     load_lambo = False

# from m2bo_bench.problem.lambo import *
# # assert 0, load_lambo
# load_chem = True
# try:
#     from m2bo_bench.problem.moo_molecule_funcs import *
# except:
#     load_chem = False

# from m2bo_bench.problem.moo_molecule_funcs import *

# def get_problem(env_name, *args, **kwargs):
#     env_name = env_name.lower()

#     PROBLEMS = {
#         'vlmop1': VLMOP1,
#         'vlmop2': VLMOP2,
#         'vlmop3': VLMOP3,
#         'kursawe': Kursawe,
#         'omnitest': OmniTest,
#         'sympart': SYMPARTRotated,
#         'zdt1': ZDT1,
#         'zdt2': ZDT2,
#         'zdt3': ZDT3,
#         'zdt4': ZDT4,
#         'zdt6': ZDT6,
#         'dtlz1': DTLZ1,
#         'dtlz2': DTLZ2,
#         'dtlz3': DTLZ3,
#         'dtlz4': DTLZ4,
#         'dtlz5': DTLZ5,
#         'dtlz6': DTLZ6,
#         'dtlz7': DTLZ7,
#         're21': RE21,
#         're22': RE22,
#         're23': RE23,
#         're24': RE24,
#         're25': RE25,
#         're31': RE31,
#         're32': RE32,
#         're33': RE33,
#         're34': RE34,
#         're35': RE35,
#         're36': RE36,
#         're37': RE37,
#         're41': RE41,
#         're42': RE42,
#         're61': RE61,
#         're91': RE91,
#         # 'qm9': QM9,
#         'motsp_500': MOTSP_500,
#         'motsp_100': MOTSP_100,
#         'motsp_50': MOTSP_50,
#         'motsp_20': MOTSP_20,
#         'mokp_200': MOKP_200,
#         'mokp_100': MOKP_100,
#         'mokp_50': MOKP_50,
#         'mocvrp_100': MOCVRP_100,
#         'mocvrp_50': MOCVRP_50,
#         'mocvrp_20': MOCVRP_20,
#         'motsp3obj_500': MOTSP3obj_500,
#         'motsp3obj_100': MOTSP3obj_100,
#         'motsp3obj_50': MOTSP3obj_50,
#         'motsp3obj_20': MOTSP3obj_20
#         # 'mo_nas': MO_NAS,
#     #     'mo_hopper_v2': MO_Hopper_V2,
#     #     'mo_swimmer_v2': MO_Swimmer_V2,
#     }

#     if load_nas:
#         PROBLEMS['nb201_test'] = NB201_Test
        
#         PROBLEMS['c10mop1'] = C10MOP1
#         PROBLEMS['c10mop2'] = C10MOP2
#         PROBLEMS['c10mop3'] = C10MOP3
#         PROBLEMS['c10mop4'] = C10MOP4
#         PROBLEMS['c10mop5'] = C10MOP5
#         PROBLEMS['c10mop6'] = C10MOP6
#         PROBLEMS['c10mop7'] = C10MOP7
#         PROBLEMS['c10mop8'] = C10MOP8
#         PROBLEMS['c10mop9'] = C10MOP9

#         PROBLEMS['in1kmop1'] = IN1KMOP1
#         PROBLEMS['in1kmop2'] = IN1KMOP2
#         PROBLEMS['in1kmop3'] = IN1KMOP3
#         PROBLEMS['in1kmop4'] = IN1KMOP4
#         PROBLEMS['in1kmop5'] = IN1KMOP5
#         PROBLEMS['in1kmop6'] = IN1KMOP6
#         PROBLEMS['in1kmop7'] = IN1KMOP7
#         PROBLEMS['in1kmop8'] = IN1KMOP8
#         PROBLEMS['in1kmop9'] = IN1KMOP9

#     if load_morl:
#         PROBLEMS['mo_hopper_v2'] = MO_Hopper_V2
#         PROBLEMS['mo_swimmer_v2'] = MO_Swimmer_V2

#     if load_chem:
#         PROBLEMS['molecule'] = Molecule
    
#     if load_lambo:
#         PROBLEMS['regex'] = REGEX
#         PROBLEMS['rfp'] = RFP
#         PROBLEMS['zinc'] = ZINC

#     if env_name not in PROBLEMS.keys():
#         raise Exception(f"Problem {env_name} not found or an importing error occurred.")
    
#     return PROBLEMS[env_name](*args, **kwargs)
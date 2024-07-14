from .mobo_vallina import MOBOVallina
from .mobo_parego import MOBOParEGO

def get_mobo_solver(solver_type: str):
    solver_type = solver_type.lower()
    type2solver = {
        'vallina': MOBOVallina,
        'parego': MOBOParEGO,
    }
    assert solver_type in type2solver.keys(), f"MOBO solver {solver_type} not found"
    return type2solver[solver_type]
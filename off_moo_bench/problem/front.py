def get_problem(name, *args, **kwargs):
    name = name.lower()
    from pymoo.problems.many import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
    from pymoo.problems.multi import ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6

    PROBLEM = {
        "dtlz1": DTLZ1,
        "dtlz2": DTLZ2,
        "dtlz3": DTLZ3,
        "dtlz4": DTLZ4,
        "dtlz5": DTLZ5,
        "dtlz6": DTLZ6,
        "dtlz7": DTLZ7,
        "zdt1": ZDT1,
        "zdt2": ZDT2,
        "zdt3": ZDT3,
        "zdt4": ZDT4,
        "zdt5": ZDT5,
        "zdt6": ZDT6,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)

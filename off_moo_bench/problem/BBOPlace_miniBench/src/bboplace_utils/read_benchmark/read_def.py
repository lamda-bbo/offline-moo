import os
import re
from copy import deepcopy

import numpy as np
from bboplace_utils.read_benchmark.rules import (
    AlwaysDismatchedRule,
    AlwaysMatchedRule,
    EntryFormatWithRuleGroups,
    RegExRule,
    RuleGroup,
    SkipRule,
)


def read_benchmark(design_name, benchmark_path, n_macro):
    database = {}

    node_info = {}
    node_info_raw_id_name = {}
    node_cnt = 0
    port_info = {}
    cell_name = []
    port_to_net_dict = {}

    def_path = os.path.join(benchmark_path, f"{design_name}.def")
    read_def(def_path, database)

    lef_path = os.path.join(benchmark_path, f"{design_name}.lef")
    read_lef(lef_path, database)

    # compute area each macro type
    macro_type_area = {}
    for macro_type in database["macro_size"]:
        size_x, size_y = database["macro_size"][macro_type]
        area = size_x * size_y
        macro_type_area[macro_type] = area

    # compute area each node
    node_area_dict = {}
    node_total_area = 0
    for node in database["nodes"]:
        node_type = database["nodes"][node]["node_type"]
        area = macro_type_area[node_type]
        node_area_dict[node] = area
        node_total_area += area

    node_lst = list(node_area_dict.keys())
    node_lst = sorted(node_lst, key=lambda x: node_area_dict[x], reverse=True)

    assert len(node_lst) >= n_macro
    macro_lst = node_lst[:n_macro]
    cell_name = node_lst[n_macro:]

    scale_factor = database["scale_factor"]
    canvas_lx = database["diearea_rect"][0] / scale_factor
    canvas_ly = database["diearea_rect"][1] / scale_factor
    canvas_ux = database["diearea_rect"][2] / scale_factor
    canvas_uy = database["diearea_rect"][3] / scale_factor
    for id, macro in enumerate(macro_lst):
        # scaling
        place_x = eval(database["nodes"][macro]["x"]) / scale_factor
        place_y = eval(database["nodes"][macro]["y"]) / scale_factor

        macro_type = database["nodes"][macro]["node_type"]
        size_x, size_y = database["macro_size"][macro_type]

        node_info[macro] = {
            "id": id,
            "size_x": size_x,
            "size_y": size_y,
            "area": size_x * size_y,
        }
        node_info[macro]["raw_x"] = place_x
        node_info[macro]["raw_y"] = place_y
        node_info_raw_id_name[id] = macro

    node_cnt = len(node_info)
    assert node_cnt == n_macro

    v_path = os.path.join(benchmark_path, f"{design_name}.v")
    net_info = read_v(v_path, node_info, database)

    net_cnt = len(net_info)

    benchmark_info = {
        "node_info": node_info,
        "node_info_raw_id_name": node_info_raw_id_name,
        "node_cnt": node_cnt,
        "port_info": port_info,
        "net_info": net_info,
        "net_cnt": net_cnt,
        "canvas_lx": canvas_lx,
        "canvas_ly": canvas_ly,
        "canvas_ux": canvas_ux,
        "canvas_uy": canvas_uy,
        "cell_name": cell_name,
        "port_to_net_dict": port_to_net_dict,
        "node_total_area": node_total_area,
        "database": database,
    }
    return benchmark_info


def read_def(path, database: dict):
    component_start_rule = RegExRule(
        r"COMPONENTS\s+(\d+)\s*;",
        {
            "num_comps": 1,
        },
    )
    component_end_rule = RegExRule(r"END\s+COMPONENTS", {})
    design_rule = RegExRule(
        r"\s*DESIGN\s+([a-z_A-Z]+)\s+([a-z_A-Z]+)\s+(\d+\.?\d+)\s*;\s*\n?",
        {
            "entry": 0,
            "key": 1,
            "value": 3,
        },
    )
    diearea_rule = RegExRule(
        r"DIEAREA\s+\(\s*(\d+)\s*(\d+)\s*\)\s*\(\s*(\d+)\s*(\d+)\s*\)\s*;\s*\n?",
        {
            "lower_x": 1,
            "lower_y": 2,
            "upper_x": 3,
            "upper_y": 4,
        },
    )
    scale_factor_rule = RegExRule(
        r"UNITS\s+DISTANCE\s+MICRONS\s(\d+)", {"entry": 0, "scale_factor": 1}
    )
    component_rule_group = RuleGroup(
        component_start_rule,
        component_end_rule,
        [
            component_start_rule,
            component_end_rule,
            RegExRule(
                r"(-)\s+(\w+)\s+(\w+)",
                {
                    "head": 1,
                    "node_name": 2,
                    "node_type": 3,
                },
            ),
            RegExRule(
                r"(\+)\s+(\w+)\s*\(\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*\)\s*(\w+)\s+;",
                {
                    "head": 1,
                    "state": 2,
                    "x": 3,
                    "y": 7,
                    "dir": 11,
                },
            ),
        ],
    )

    other_rule_group = RuleGroup(
        AlwaysMatchedRule(),
        AlwaysMatchedRule(),
        [scale_factor_rule, design_rule, diearea_rule, SkipRule()],
    )

    def_ent_format = EntryFormatWithRuleGroups([component_rule_group, other_rule_group])

    assert path is not None and os.path.exists(path)
    database["nodes"] = {}
    database["def_origin"] = [[]]
    database["scale_factor"] = None
    database["design_config"] = {}
    database["diearea_rect"] = []
    with open(path, "r") as f:
        for line in f:
            output = def_ent_format(line)
            if output.get("entry", None) is not None:
                entry = output.get("entry")
                database["def_origin"][-1].append("%s" % (entry))
            else:
                database["def_origin"].append([])
            if output == {}:
                continue

            if "num_comps" in output.keys():
                database.update(output)
            elif "scale_factor" in output.keys():
                database["scale_factor"] = int(output["scale_factor"])
            elif "key" in output.keys():
                database["design_config"][output["key"]] = eval(output["value"])
            elif "lower_x" in output.keys():
                database["diearea_rect"].extend(
                    [
                        eval(output["lower_x"]),
                        eval(output["lower_y"]),
                        eval(output["upper_x"]),
                        eval(output["upper_y"]),
                    ]
                )
            else:
                head = output.get("head", None)
                if head == "-":
                    node_name = output["node_name"]
                    database["nodes"][node_name] = deepcopy(output)
                elif head == "+":
                    database["nodes"][node_name].update(output)
                else:
                    continue

    for i in range(len(database["def_origin"])):
        database["def_origin"][i] = "".join(database["def_origin"][i])


def read_lef(path, database: dict):
    macro_start_rule = RegExRule(
        r"MACRO\s+(\w+)",
        {
            "macro_name": 1,
        },
    )
    macro_or_pin_end_rule = RegExRule(
        r"END\s+(\w+)",
        {
            "macro_name_or_pin_name": 1,
        },
    )
    macro_size_rule = RegExRule(
        r"SIZE\s(\d+(\.\d+)?) BY (\d+(\.\d+)?)",
        {
            "size_x": 1,
            "size_y": 3,
        },
    )
    macro_pin_start_rule = RegExRule(
        r"(PIN)\s+(\w+)",
        {
            "pin_start": 1,
            "pin_name": 2,
        },
    )
    macro_pin_offset_rule = RegExRule(
        r"RECT\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)\s(-?\d+(\.\d+)?)",
        {
            "x1": 1,
            "y1": 3,
            "x2": 5,
            "y2": 7,
        },
    )

    macro_rule_group = RuleGroup(
        macro_start_rule,
        AlwaysDismatchedRule(),
        [
            macro_start_rule,
            macro_or_pin_end_rule,
            macro_size_rule,
            macro_pin_start_rule,
            macro_pin_offset_rule,
            RegExRule(
                r"CLASS\s+(\w+)\s+;",
                {
                    "class": 1,
                },
            ),
            SkipRule(),
        ],
    )

    other_rule_group = RuleGroup(AlwaysMatchedRule(), AlwaysMatchedRule(), [SkipRule()])

    lef_ent_format = EntryFormatWithRuleGroups([macro_rule_group, other_rule_group])

    assert path is not None and os.path.exists(path), path
    database["lef_macros"] = {}
    database["lef_origin"] = [[]]
    database["macro_size"] = {}
    database["pin_offset"] = {}
    macro_name = None
    pin_name = None
    pin_flag = False
    with open(path, "r") as f:
        for line in f:
            output = lef_ent_format(line)
            if not lef_ent_format.inGroup():
                database["lef_origin"][-1].append(line)
                continue

            if macro_name is None:
                macro_name = output.get("macro_name", None)
                if macro_name is not None:
                    database["lef_macros"][macro_name] = line
                    database["lef_origin"].append([])
                    database["pin_offset"][macro_name] = {}
                continue

            # print(output.keys())
            if "macro_name_or_pin_name" in output.keys():
                if pin_flag:
                    pin_flag = False

                    min_pin_grid_x = np.min(pin_grid_x)
                    max_pin_grid_x = np.max(pin_grid_x)
                    min_pin_grid_y = np.min(pin_grid_y)
                    max_pin_grid_y = np.max(pin_grid_y)
                    pin_name = output["macro_name_or_pin_name"]
                    database["pin_offset"][macro_name][pin_name] = (
                        (min_pin_grid_x + max_pin_grid_x) / 2,
                        (min_pin_grid_y + max_pin_grid_y) / 2,
                    )
                    del pin_grid_x
                    del pin_grid_y
                    pin_flag = False
                else:
                    database["lef_macros"][macro_name] += line
                    if output["macro_name_or_pin_name"] == macro_name:
                        lef_ent_format.quitGroup()
                        macro_name = None
            elif "size_x" in output.keys():
                database["macro_size"][macro_name] = (
                    eval(output["size_x"]),
                    eval(output["size_y"]),
                )
            elif "class" in output.keys():
                # rule: CLASS (CORE|BLOCK)
                c = output["class"]
                if c == "CORE":
                    database["lef_macros"][macro_name] += line
                else:
                    database["lef_macros"][macro_name] += line.replace(c, "CORE", 1)
            elif "pin_start" in output.keys():
                pin_flag = True
                pin_grid_x = []
                pin_grid_y = []
            elif "x1" in output.keys():
                if pin_flag:
                    x1, y1 = eval(output["x1"]), eval(output["y1"])
                    x2, y2 = eval(output["x2"]), eval(output["y2"])
                    pin_grid_x.append(x1)
                    pin_grid_x.append(x2)
                    pin_grid_y.append(y1)
                    pin_grid_y.append(y2)
            else:
                # rule: skip
                database["lef_macros"][macro_name] += line

    for i in range(len(database["lef_origin"])):
        database["lef_origin"][i] = "".join(database["lef_origin"][i])


def read_v(path, node_info, database):
    with open(path, "r") as f:
        content = f.readlines()
    net_info = {}
    net_cnt = 0
    flag = 0
    for line in content:
        if "wire" in line:
            line_ls = line.split(" ")
            net_name = line_ls[1].split(";")[0]
            net_info[net_name] = {}
            net_info[net_name]["nodes"] = {}
            net_info[net_name]["ports"] = {}

        if "Start cells" in line:
            flag = 1
            continue
        if flag == 1:
            if line == "\n":
                break
            pattern = r"\.(\w+)\((.*?)\)"
            matches = re.findall(pattern=pattern, string=line)
            line_ls = line.split(" ")
            node_type = line_ls[0]
            node_name = line_ls[1]

            for pin_net in matches:
                pin = pin_net[0]
                net = pin_net[1]
                if node_name in node_info.keys() and net in net_info.keys():
                    x_offset, y_offset = database["pin_offset"][node_type][pin]
                    net_info[net]["nodes"][node_name] = {
                        "x_offset": x_offset,
                        "y_offset": y_offset,
                    }

    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]["id"] = net_cnt
        net_cnt += 1
    return net_info

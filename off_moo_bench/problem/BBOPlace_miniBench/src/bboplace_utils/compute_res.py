import heapq
import math

import numpy as np
from src.placedb import PlaceDB

from bboplace_utils.constant import INF
from bboplace_utils.debug import *


def _comp_net_hpwl(macro_pos, placedb: PlaceDB):
    assert len(macro_pos) > 0

    net_hwpl = {}
    for net_name in placedb.net_info:
        max_x = 0.0
        min_x = placedb.canvas_width * 1.1
        max_y = 0.0
        min_y = placedb.canvas_height * 1.1
        for macro in placedb.net_info[net_name]["nodes"]:
            size_x = placedb.node_info[macro]["size_x"]
            size_y = placedb.node_info[macro]["size_y"]
            pin_x = (
                macro_pos[macro][0]
                + size_x / 2
                + placedb.net_info[net_name]["nodes"][macro]["x_offset"]
            )
            pin_y = (
                macro_pos[macro][1]
                + size_y / 2
                + placedb.net_info[net_name]["nodes"][macro]["y_offset"]
            )
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)

        net_hwpl[net_name] = (min_x, min_y, max_x, max_y)

    return net_hwpl


def comp_res(macro_pos: dict, placedb: PlaceDB):
    if len(macro_pos) == 0:
        return INF, INF, INF
    net_hpwl = _comp_net_hpwl(macro_pos=macro_pos, placedb=placedb)
    hpwl = _comp_res_hpwl(net_hpwl, placedb)
    congestion = _comp_res_congestion(net_hpwl, placedb)
    # congestion_old = _comp_res_congestion_old(net_hpwl, placedb)
    regularity = _comp_res_regularity(macro_pos, placedb)
    return hpwl, congestion, regularity


def _comp_res_hpwl(net_hpwl: dict, placedb: PlaceDB):
    hpwl = 0.0
    for net_name, bounding_box in net_hpwl.items():
        min_x, min_y, max_x, max_y = bounding_box
        hpwl_temp = (max_x - min_x) + (max_y - min_y)

        if "weight" in placedb.net_info[net_name]:
            hpwl_temp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_temp
    return hpwl


def _comp_res_congestion_old(net_hpwl: dict, placedb: PlaceDB):
    congestion = np.zeros((placedb.canvas_width, placedb.canvas_height))

    for min_x, min_y, max_x, max_y in net_hpwl.values():
        min_x = max(0, math.ceil(min_x))
        min_y = max(0, math.ceil(min_y))
        max_x = min(placedb.canvas_width, math.ceil(max_x))
        max_y = min(placedb.canvas_height, math.ceil(max_y))
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        if delta_x == 0 or delta_y == 0:
            continue

        congestion[min_x:max_x, min_y:max_y] += 1 / delta_x + 1 / delta_y

    congestion_list = congestion.reshape(1, -1).tolist()[0]
    congestion_mean = np.mean(
        heapq.nlargest(math.ceil(len(congestion_list) / 10), congestion_list)
    )
    return congestion_mean

def _comp_res_congestion(net_hpwl: dict, placedb: PlaceDB):
    congestion = np.zeros((placedb.canvas_width, placedb.canvas_height), dtype=np.float32)
    
    coords = np.array(list(net_hpwl.values()))
    if len(coords) == 0:
        return 0.0
        
    coords[:, 0] = np.maximum(0, np.ceil(coords[:, 0]))  # min_x
    coords[:, 1] = np.maximum(0, np.ceil(coords[:, 1]))  # min_y
    coords[:, 2] = np.minimum(placedb.canvas_width, np.ceil(coords[:, 2]))   # max_x
    coords[:, 3] = np.minimum(placedb.canvas_height, np.ceil(coords[:, 3]))  # max_y
    
    delta_x = coords[:, 2] - coords[:, 0]
    delta_y = coords[:, 3] - coords[:, 1]
    
    valid_nets = (delta_x > 0) & (delta_y > 0)
    coords = coords[valid_nets]
    delta_x = delta_x[valid_nets]
    delta_y = delta_y[valid_nets]
    
    for i in range(len(coords)):
        min_x, min_y, max_x, max_y = coords[i].astype(int)
        congestion[min_x:max_x, min_y:max_y] += 1/delta_x[i] + 1/delta_y[i]
    
    k = max(1, int(congestion.size * 0.1))
    return np.partition(congestion.ravel(), -k)[-k:].mean()

def _comp_res_regularity(macro_pos: dict, placedb: PlaceDB):
    # FIXME: Warning, regularity is not feasible for sequence pair formulation
    # since macro will exceed the chip canvas
    x_dis_from_edge = 0
    y_dis_from_edge = 0
    total_area = 0
    for macro_name, (macro_lx, macro_ly) in macro_pos.items():
        macro_ux = macro_lx + placedb.node_info[macro_name]["size_x"]
        macro_uy = macro_ly + placedb.node_info[macro_name]["size_y"]
        area = placedb.node_info[macro_name]["area"]
        total_area += area

        x_dis_from_edge += min(macro_lx, max(placedb.canvas_width - macro_ux, 0)) * area
        y_dis_from_edge += (
            min(macro_ly, max(placedb.canvas_height - macro_uy, 0)) * area
        )

    return (x_dis_from_edge + y_dis_from_edge) / total_area

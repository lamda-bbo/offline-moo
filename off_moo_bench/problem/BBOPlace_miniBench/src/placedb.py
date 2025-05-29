import datetime
import logging

from bboplace_utils.read_benchmark import REGISTRY as READ_REGISTRY


class PlaceDB:
    def __init__(self, args):
        self.args = args

        logging.info("reading benchmark from {}".format(args.benchmark_path))
        benchmark_info = READ_REGISTRY[args.benchmark_base](
            design_name=args.design_name,
            benchmark_path=args.benchmark_path,
            n_macro=args.n_macro,
        )

        self.node_info = benchmark_info["node_info"]
        self.node_info_raw_id_name = benchmark_info["node_info_raw_id_name"]
        self.node_cnt = benchmark_info["node_cnt"]
        self.port_info = benchmark_info["port_info"]
        self.net_info = benchmark_info["net_info"]
        self.net_cnt = benchmark_info["net_cnt"]
        self.canvas_lx = benchmark_info["canvas_lx"]
        self.canvas_ly = benchmark_info["canvas_ly"]
        self.canvas_ux = benchmark_info["canvas_ux"]
        self.canvas_uy = benchmark_info["canvas_uy"]
        self.cell_name = benchmark_info["cell_name"]
        self.port_to_net_dict = benchmark_info["port_to_net_dict"]
        self.node_total_area = benchmark_info["node_total_area"]
        self.database = benchmark_info["database"]
        self.node_to_net_dict = get_node_to_net_dict(
            node_info=self.node_info, net_info=self.net_info
        )

        self.canvas_width = self.canvas_ux - self.canvas_lx
        self.canvas_height = self.canvas_uy - self.canvas_ly
        self.macro_lst = list(self.node_info.keys())

        args.n_macro = self.node_cnt

    def to_pl(self, macro_pos=None, fix_macro=True) -> str:
        if macro_pos is None:
            macro_pos = {
                macro: (
                    self.node_info[macro]["raw_x"],
                    self.node_info[macro]["raw_y"],
                )
                for macro in self.macro_lst
            }

        content = ""
        content += "UCLA pl 1.0\n"
        content += "# Created\t:\t%s\n\n" % datetime.datetime.now().strftime("%b %d %Y")
        for std_cell in self.cell_name:
            content += "{}\t{}\t{}\t:\tN\n".format(std_cell, 0, 0)

        if fix_macro:
            fixed = "/FIXED"
        else:
            fixed = ""

        for macro in self.macro_lst:
            try:
                x, y = macro_pos[macro]
            except:
                assert 0, macro_pos
            content += "{}\t{}\t{}\t:\tN {}\n".format(
                macro, round(x + self.canvas_lx), round(y + self.canvas_ly), fixed
            )
        return content

    def to_def(self, macro_pos=None, fix_macro=True) -> str:
        if macro_pos is None:
            macro_pos = {}
        def_origin = list(reversed(self.database["def_origin"]))

        content = "###############################################################\n"
        content += "#  Generated on\t:\t%s\n" % datetime.datetime.now().strftime(
            "%b %d %Y"
        )
        content += "###############################################################\n"

        content += def_origin.pop()
        content += "DIEAREA ( {} {} ) ( {} {} ) ;\n".format(
            *tuple(self.database["diearea_rect"])
        )
        content += def_origin.pop()

        content += "COMPONENTS %s ;\n" % (self.database["num_comps"])

        node_list = self.database["nodes"].keys()
        inv_ratio_x = self.database["scale_factor"]
        inv_ratio_y = inv_ratio_x
        for node_name in node_list:
            node_info = self.database["nodes"][node_name]
            content += "- %(node_name)s %(node_type)s\n" % node_info
            if node_name in macro_pos.keys():
                x, y = macro_pos[node_name]

                # inv scaling
                x = round(x * inv_ratio_x + self.canvas_lx)
                y = round(y * inv_ratio_y + self.canvas_ly)

                status = "FIXED" if fix_macro else "PLACED"

                content += f"\t+ {status} ( {x} {y} ) {node_info['dir']} ;\n"
            else:
                content += "\t+ PLACED ( %(x)s %(y)s ) %(dir)s ;\n" % node_info

        content += "END COMPONENTS\n"

        while len(def_origin) > 0:
            content += def_origin.pop()

        return content


def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict

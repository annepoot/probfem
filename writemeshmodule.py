import numpy as np
import os
from scipy.integrate import quad

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import check_dict, check_list
from myjive.util import to_xtable


class WriteMeshModule(Module):
    def WRITEMESH(self, globdat, fname, ftype, **kwargs):
        if "manual" in ftype:
            self._write_mesh(globdat, fname)
        else:
            raise ValueError("Invalid file type passed to WRITEMESH")

    def _write_mesh(self, globdat, fname):
        nodes = globdat[gn.NSET]
        elems = globdat[gn.ESET]

        path, file = os.path.split(fname)
        if len(path) > 0 and not os.path.isdir(path):
            os.makedirs(path)

        with open(fname, "w") as file:
            file.write("nodes (ID, x, [y], [z])\n")
            for inode, node in enumerate(nodes):
                node_id = nodes.get_node_id(inode)
                coords = node.get_coords()
                file.write("{} ".format(node_id))
                file.write(" ".join(["{}".format(coord) for coord in coords]))
                file.write("\n")

            file.write("elements (node#1, node#2, [node#3, ...])\n")
            for ielem, elem in enumerate(elems):
                inodes = elems.get_elem_nodes(ielem)
                node_ids = nodes.get_node_ids(inodes)
                file.write(" ".join(["{}".format(node_id) for node_id in node_ids]))
                file.write("\n")

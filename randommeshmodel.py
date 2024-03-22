import numpy as np
import os

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import optarg


class RandomMeshModel(Model):
    def PERTURBNODES(self, nodes, globdat, rng=np.random.default_rng(), **kwargs):
        nodes = self._perturb_nodes(nodes, globdat, rng=rng)
        return nodes

    def WRITEMESH(self, globdat, fname, ftype, **kwargs):
        if "manual" in ftype:
            self._write_mesh(globdat, fname)
        else:
            raise ValueError("Invalid file type passed to WRITEMESH")

    def configure(self, globdat, **props):
        # get props
        pass

    def _perturb_nodes(self, nodes, globdat, rng=np.random.default_rng()):
        for node in nodes:
            coords = node.get_coords()
            coords += rng.uniform(-0.05, 0.05)
            node.set_coords(coords)

        return nodes

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

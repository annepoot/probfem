import numpy as np
import os

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import mdtlist, mdtdict, mdtarg


class RandomMeshModel(Model):
    def PERTURBNODES(
        self, nodes, globdat, meshsize, rng=np.random.default_rng(), **kwargs
    ):
        nodes = self._perturb_nodes(nodes, globdat, meshsize, rng=rng)
        return nodes

    def WRITEMESH(self, globdat, fname, ftype, **kwargs):
        if "manual" in ftype:
            self._write_mesh(globdat, fname)
        else:
            raise ValueError("Invalid file type passed to WRITEMESH")

    def configure(self, globdat, **props):
        # get props
        self._p = mdtarg(self, props, "p", dtype=float)
        bprops = mdtdict(self, props, "boundary", ["groups"])

        self._bgroups = mdtlist(self, bprops, "groups")
        bnodes = set()
        for group in self._bgroups:
            ngroup = globdat[gn.NGROUPS][group]
            for inode in ngroup:
                bnodes.add(inode)
        self._bnodes = list(bnodes)

    def _perturb_nodes(self, nodes, globdat, meshsize, rng=np.random.default_rng()):
        h = np.max(meshsize[""])

        for inode, node in enumerate(nodes):
            if inode not in self._bnodes:
                alpha_i_bar = rng.uniform(-0.5, 0.5)
                ielem = inode - 1
                h_i_bar = min(meshsize[""][ielem], meshsize[""][ielem + 1])
                alpha_i = (h_i_bar / h) ** self._p * alpha_i_bar

                coords = node.get_coords()
                coords += h**self._p * alpha_i

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

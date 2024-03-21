import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.fem import to_xnodeset

ELEMENTS = "elements"
EA = "EA"
k = "k"
RHOA = "rhoA"
SHAPE = "shape"
TYPE = "type"
INTSCHEME = "intScheme"
DOFTYPES = ["dx"]


class RandomMeshModel(Model):
    def PERTURBNODES(self, nodes, globdat, **kwargs):
        nodes = self._perturb_nodes(nodes, globdat)
        return nodes

    def configure(self, globdat, **props):
        pass

    def _perturb_nodes(self, nodes, globdat):

        for node in nodes:
            coords = node.get_coords()
            coords += np.random.uniform(-0.01, 0.01)
            node.set_coords(coords)

        return nodes

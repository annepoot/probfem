import numpy as np
import matplotlib.pyplot as plt

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import mdtarg, mdtdict, optarg
from copy import deepcopy


class RMPlotModule(Module):
    def init(self, globdat, **props):

        self._figprops = optarg(self, props, "figure", dtype=dict)
        self._refprops = optarg(self, props, "reference", dtype=dict)
        self._pertprops = optarg(self, props, "perturbed", dtype=dict)

    def run(self, globdat):

        # Get reference solve
        x = self._get_x(globdat)
        u = globdat[gn.STATE0]

        # Get perturbed solves
        x_pert = []
        u_pert = []
        for globdat_pert in globdat["perturbedSolves"]:
            xp = self._get_x(globdat_pert)
            up = globdat_pert[gn.STATE0]
            x_pert.append(xp)
            u_pert.append(up)

        # Plot both in a single figure
        fig, ax = plt.subplots(1, 1)
        for xp, up in zip(x_pert, u_pert):
            ax.plot(xp, up, **self._pertprops)
        ax.plot(x, u, **self._refprops)
        ax.set(**self._figprops)
        plt.show()

        return "ok"

    def _get_x(self, globdat):
        nodes = globdat[gn.NSET]
        dofs = globdat[gn.DOFSPACE]

        dc = dofs.dof_count()
        x = np.zeros(dc)

        doftypes = dofs.get_types()
        if len(doftypes) > 1:
            raise RuntimeError("RMPlotModule has only been implemented in 1D")
        doftype = list(dofs.get_types())[0]

        for inode, node in enumerate(nodes):
            idof = dofs.get_dof(inode, doftype)
            coords = node.get_coords()
            x[idof] = coords[0]

        return x

    def shutdown(self, globdat):
        pass

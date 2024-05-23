import numpy as np
import matplotlib.pyplot as plt

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.fem import XPointSet
from myjive.util import Table
from myjive.util.proputils import check_value, get_recursive, split_key


class RMPlotModule(Module):
    @Module.save_config
    def configure(
        self,
        globdat,
        *,
        field,
        comp,
        plotType,
        figure={},
        exact={},
        fem={},
        femField=None,
        perturbed={},
        save={}
    ):
        # Validate input arguments
        check_value(self, plotType, ["node", "elem"])
        self._field = field
        self._comp = comp
        self._plottype = plotType
        self._figprops = figure
        self._exactprops = exact
        self._femprops = fem
        self._femfield = femField
        self._pertprops = perturbed
        self._saveprops = save

    def init(self, globdat):
        pass

    def run(self, globdat):
        # Get the exact solution
        x_exact, field_exact = self._get_exact_solution(globdat)

        # Get the FEM solution
        x_fem, field_fem = self._get_fem_solution(globdat)

        # Get perturbed solves
        x_pert = []
        field_pert = []
        for globdat_pert in globdat["perturbedSolves"]:
            xp, fieldp = self._get_fem_solution(globdat_pert)
            x_pert.append(xp)
            field_pert.append(fieldp)

        # Plot all in a single figure
        fig, ax = plt.subplots(1, 1)
        for xp, fieldp in zip(x_pert, field_pert):
            if self._plottype == "node":
                ax.plot(xp, fieldp, **self._pertprops)
            elif self._plottype == "elem":
                ax.step(xp[1:], fieldp, **self._pertprops)
            else:
                assert False

        ax.plot(x_exact, field_exact, **self._exactprops)

        if self._plottype == "node":
            ax.plot(x_fem, field_fem, **self._femprops)
        elif self._plottype == "elem":
            ax.step(x_fem[1:], field_fem, **self._femprops)
        else:
            assert False

        ax.set(**self._figprops)

        if len(self._saveprops) > 0:
            plt.savefig(**self._saveprops)
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

    def _get_exact_solution(self, globdat):
        points = XPointSet()
        for coord in np.linspace(0, 1, 1000):
            points.add_point([coord])
        points.to_pointset()

        table = Table(size=len(points))
        models = self.get_relevant_models("GETEXACTTABLE", globdat[gn.MODELS])
        for model in models:
            table = model.GETEXACTTABLE(self._field, table, globdat, points)

        field_exact = table[self._comp]
        x_exact = points.get_coords().flatten()

        return x_exact, field_exact

    def _get_fem_solution(self, globdat):
        if self._femfield is None:
            if self._plottype == "node":
                nodecount = len(globdat[gn.NSET])
                table = Table(size=nodecount)
                tbwts = np.zeros(nodecount)
                models = self.get_relevant_models("GETTABLE", globdat[gn.MODELS])
                for model in models:
                    table, tbwts = model.GETTABLE(self._field, table, tbwts, globdat)
            elif self._plottype == "elem":
                table = Table(size=len(globdat[gn.ESET]))
                models = self.get_relevant_models("GETELEMTABLE", globdat[gn.MODELS])
                for model in models:
                    table = model.GETELEMTABLE(self._field, table, globdat)
            else:
                assert False

            field_fem = table[self._comp]
        else:
            field_fem = get_recursive(globdat, split_key(self._femfield))

        x_fem = self._get_x(globdat)

        return x_fem, field_fem

    def shutdown(self, globdat):
        pass

import os
import numpy as np
import matplotlib.pyplot as plt

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util import Table, to_xtable
from myjive.util.proputils import split_key, get_recursive
from myjivex.modules.viewmodule import generate_plot


class ProbViewModule(Module):
    @Module.save_config
    def configure(
        self,
        globdat,
        *,
        tables=[],
        elemTables=[],
        keys=[],
        scale=0.0,
        line={},
        fill={},
        colorbar={},
        save={},
        figure={},
        axes={}
    ):
        # Validate input arguments
        self._ntables = tables
        self._etables = elemTables
        self._keys = keys
        self._scale = scale
        self._lineprops = line
        self._fillprops = fill
        self._cbarprops = colorbar
        self._saveprops = save
        self._figprops = figure
        self._axprops = axes

    def init(self, globdat):
        pass

    def run(self, globdat):
        cglobdat = globdat["fine"]
        fglobdat = globdat["fine"]

        for name in self._ntables:
            self._write_node_table(name, cglobdat)
            self._write_node_table(name, fglobdat)

        for name in self._etables:
            self._write_elem_table(name, cglobdat)
            self._write_elem_table(name, fglobdat)

        return "ok"

    def shutdown(self, globdat):
        fglobdat = globdat["fine"]

        fig, axs = plt.subplots(
            nrows=len(self._keys), ncols=1, squeeze=False, **self._figprops
        )

        for i, key in enumerate(self._keys):
            field = get_recursive(globdat, split_key(key))
            generate_plot(
                field,
                axs[i, 0],
                fglobdat,
                plottype="node",
                scale=self._scale,
                fillprops=self._fillprops,
                lineprops=self._lineprops,
                cbarprops=self._cbarprops,
                axprops=self._axprops,
            )

        if "fname" in self._saveprops:
            fname = self._saveprops["fname"]
            dirname = os.path.dirname(fname)
            if len(dirname) > 0 and not os.path.isdir(dirname):
                os.makedirs(dirname)
            plt.savefig(fname, *self._saveprops)

        plt.show()

    def _write_node_table(self, name, globdat):
        if name not in globdat[gn.TABLES]:
            nodecount = len(globdat[gn.NSET])
            table = Table(size=nodecount)
            tbwts = np.zeros(nodecount)

            for model in self.get_relevant_models("GETTABLE", globdat[gn.MODELS]):
                table, tbwts = model.GETTABLE(name, table, tbwts, globdat)

            to_xtable(table)

            for jcol in range(table.column_count()):
                values = table.get_col_values(None, jcol)
                table.set_col_values(None, jcol, values / tbwts)

            table.to_table()
            globdat[gn.TABLES][name] = table

    def _write_elem_table(self, name, globdat):
        if name not in globdat[gn.ELEMTABLES]:
            elemcount = len(globdat[gn.ESET])
            table = Table(size=elemcount)

            for model in self.get_relevant_models("GETELEMTABLE", globdat[gn.MODELS]):
                table = model.GETELEMTABLE(name, table, globdat)

            globdat[gn.ELEMTABLES][name] = table

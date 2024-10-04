import os
import numpy as np
import matplotlib.pyplot as plt

from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util import Table, to_xtable
from myjive.util.proputils import split_key, get_recursive, set_recursive
from myjivex.modules.viewmodule import generate_plot


class ConversionModule(Module):
    @Module.save_config
    def configure(self, globdat, sources, targets, convTypes):
        self._sources = sources
        self._targets = targets
        self._conv_types = convTypes  # field2table, coarse2fine

    def init(self, globdat):
        pass

    def run(self, globdat):
        for i, conv_type in enumerate(self._conv_types):
            source_keys = split_key(self._sources[i])
            target_keys = split_key(self._targets[i])

            if conv_type == "field2table":
                table = self._convert_field_to_table(source_keys, target_keys, globdat)
                set_recursive(globdat, target_keys, table)
            elif conv_type == "coarse2fine":
                table = self._convert_coarse_to_fine(source_keys, target_keys, globdat)
                set_recursive(globdat, target_keys, table)
            elif conv_type == "coarse2error":
                table = self._convert_coarse_to_error(source_keys, target_keys, globdat)
                set_recursive(globdat, target_keys, table)

        return "ok"

    def shutdown(self, globdat):
        pass

    def _convert_field_to_table(self, source_keys, target_keys, globdat):
        if target_keys[-2] == "tables":
            tglobdat = get_recursive(globdat, target_keys[:-2])
            if "tables" in tglobdat:
                tbls = tglobdat["tables"]
                name = target_keys[-1]
                if name in tbls:
                    table = tbls[name]
                else:
                    table = Table(size=len(tglobdat[gn.NSET]))
            else:
                tglobdat["tables"] = {}
                table = Table(size=len(tglobdat[gn.NSET]))
        else:
            raise ValueError("Expected penultimate key to be 'tables'")

        source = get_recursive(globdat, source_keys)

        elems = tglobdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = tglobdat[gn.DOFSPACE]
        dof_types = dofs.get_types()

        xtable = to_xtable(table)
        jcols = xtable.add_columns(dof_types)

        for inode in range(len(nodes)):
            idofs = dofs.get_dofs([inode], dof_types)
            xtable.set_row_values(inode, jcols, source[idofs])

        return xtable.to_table()

    def _convert_coarse_to_fine(self, source_keys, target_keys, globdat):
        if source_keys[-2] != "tables":
            raise ValueError("Expected penultimate key to be 'tables'")
        if target_keys[-2] != "tables":
            raise ValueError("Expected penultimate key to be 'tables'")

        cglobdat = get_recursive(globdat, source_keys[:-2])
        fglobdat = get_recursive(globdat, target_keys[:-2])

        if "tables" in fglobdat:
            tbls = fglobdat["tables"]
            name = target_keys[-1]
            if name in tbls:
                ftable = tbls[name]
            else:
                ftable = Table(size=len(fglobdat[gn.NSET]))
        else:
            fglobdat["tables"] = {}
            ftable = Table(size=len(fglobdat[gn.NSET]))

        ctable = get_recursive(globdat, source_keys)

        elemsc = cglobdat[gn.ESET]
        nodesc = cglobdat[gn.NSET]
        nodes = fglobdat[gn.NSET]

        xtable = to_xtable(ftable)
        jcols = xtable.add_columns(ctable.get_column_names())

        rank = fglobdat[gn.MESHRANK]
        shapefac = fglobdat[gn.SHAPEFACTORY]
        shape = shapefac.get_shape(fglobdat[gn.MESHSHAPE], "Gauss1")

        # Go over the coarse mesh
        for inodesc in elemsc:
            coordsc = nodesc.get_some_coords(inodesc)

            # Get the bounding box of the coarse element
            bbox = np.zeros((rank, 2))
            for i in range(rank):
                bbox[i, 0] = min(coordsc[:, i])
                bbox[i, 1] = max(coordsc[:, i])

            cvalues = ctable.get_block(inodesc, jcols)

            # Go over the fine mesh
            for inode, coords in enumerate(nodes):
                # Check if the node falls inside the bounding box
                inside = True
                for i in range(rank):
                    if coords[i] < bbox[i, 0] or coords[i] > bbox[i, 1]:
                        inside = False
                        break

                # If so, check if the node falls inside the shape itself
                if inside:
                    loc_point = shape.get_local_point(coords, coordsc)
                    inside = shape.contains_local_point(loc_point, tol=1e-8)

                # If so, add the relative shape function values to the Phi matrix
                if inside:
                    svals = np.round(shape.eval_shape_functions(loc_point), 12)
                    ftable.set_row_values(inode, jcols, svals @ cvalues)

        return xtable.to_table()

    def _convert_coarse_to_error(self, source_keys, target_keys, globdat):
        if source_keys[-2] != "tables":
            raise ValueError("Expected penultimate key to be 'tables'")
        if target_keys[-2] != "tables":
            raise ValueError("Expected penultimate key to be 'tables'")

        eglobdat = get_recursive(globdat, target_keys[:-2])

        ctable = get_recursive(globdat, source_keys)
        ftable = get_recursive(globdat, source_keys[:-1] + ["state0"])

        if "tables" in eglobdat:
            tbls = eglobdat["tables"]
            name = target_keys[-1]
            if name in tbls:
                etable = tbls[name]
            else:
                etable = Table(size=len(eglobdat[gn.NSET]))
        else:
            eglobdat["tables"] = {}
            etable = Table(size=len(eglobdat[gn.NSET]))

        xtable = to_xtable(etable)
        dof_types = ctable.get_column_names()
        jcols = xtable.add_columns(dof_types)

        for jcol, dof_type in zip(jcols, dof_types):
            xtable.set_col_values(None, jcol, ftable[dof_type] - ctable[dof_type])

        return xtable.to_table()

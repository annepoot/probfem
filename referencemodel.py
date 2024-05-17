import numpy as np
import sympy as sym
from scipy.integrate import quad

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util import to_xtable
import myjive.util.proputils as pu


class ReferenceModel(Model):
    def GETEXACTTABLE(self, name, table, globdat, points, **kwargs):
        if "solution" in name:
            table = self._get_exact_solution(table, points)
        elif "strain" in name:
            table = self._get_exact_strain(table, points)
        elif "stress" in name:
            table = self._get_exact_stress(table, points)
        return table

    def COMPUTEERROR(self, name, table, globdat):
        if "solution" in name:
            table = self._compute_solution_error(table, globdat)
        elif "strain" in name:
            table = self._compute_strain_error(table, globdat)
        return table

    @Model.save_config
    def configure(self, globdat, *, u, kappa, params={}):
        # Get basic dimensionality info
        self._rank = globdat[gn.MESHRANK]
        self._strcount = self._rank * (self._rank + 1) // 2

        # Get the dictionary for load evaluation
        eval_dict = self._get_sympy_eval_dict()
        eval_dict.update(params)

        # Compute the symbolic expressions
        self._u_exact = eval(u, {}, eval_dict)
        self._k_exact = eval(kappa, {}, eval_dict)

        # Get derivative info
        self._fvars = self._u_exact.free_symbols
        self._eps_exact = []
        self._sigma_exact = []
        for fvar in self._fvars:
            eps = sym.diff(self._u_exact, fvar)
            self._eps_exact.append(eps)
            sigma = self._k_exact * eps
            self._sigma_exact.append(sigma)

    def _get_exact_solution(self, table, points):
        xtable = to_xtable(table)
        jcol = xtable.add_column("dx")

        for ipoint, coords in enumerate(points):
            u = self._u_exact.subs(dict(zip(self._fvars, coords))).evalf()
            xtable.set_value(ipoint, jcol, u)

        table = xtable.to_table()
        return table

    def _get_exact_strain(self, table, points):
        xtable = to_xtable(table)
        comps = self._get_gradient_comps()
        jcols = xtable.add_columns(comps)

        for ipoint, coords in enumerate(points):
            for jcol, eps_exact in zip(jcols, self._eps_exact):
                eps = eps_exact.subs(dict(zip(self._fvars, coords))).evalf()
                xtable.set_value(ipoint, jcol, eps)

        table = xtable.to_table()
        return table

    def _get_exact_stress(self, table, points):
        xtable = to_xtable(table)
        comps = self._get_gradient_comps()
        jcols = xtable.add_columns(comps)

        for ipoint, coords in enumerate(points):
            for jcol, sigma_exact in zip(jcols, self._sigma_exact):
                sigma = sigma_exact.subs(dict(zip(self._fvars, coords))).evalf()
                xtable.set_value(ipoint, jcol, sigma)

        table = xtable.to_table()
        return table

    def _compute_solution_error(self, table, globdat):
        xtable = to_xtable(table)
        jcol = xtable.add_column("solution")

        elems = globdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = globdat[gn.DOFSPACE]

        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss999")

        str_u_exact = str(self._u_exact)
        eval_dict = self._get_numpy_eval_dict()

        for ielem, elem in enumerate(elems):
            inodes = elems.get_elem_nodes(ielem)
            idofs = dofs.get_dofs(inodes, self._get_solution_comps())
            coords = nodes.get_some_coords(inodes)

            eldisp = globdat[gn.STATE0][idofs]

            def exact_func(x):
                return pu.evaluate(str_u_exact, [x], self._rank, extra_dict=eval_dict)

            def fem_func(x):
                sfuncs = shape.eval_global_shape_functions([x], coords)
                return eldisp @ sfuncs

            def error_func(x):
                return np.sqrt((exact_func(x) - fem_func(x)) ** 2)

            norm = quad(error_func, coords[0, 0], coords[0, 1])[0]

            xtable.set_value(ielem, jcol, norm)

        table = xtable.to_table()
        return table

    def _compute_strain_error(self, table, globdat):
        xtable = to_xtable(table)

        jcol = xtable.add_column("strain")
        elems = globdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = globdat[gn.DOFSPACE]

        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss999")
        str_eps_exact = str(self._eps_exact)
        eval_dict = self._get_numpy_eval_dict()

        for ielem, elem in enumerate(elems):
            inodes = elems.get_elem_nodes(ielem)
            idofs = dofs.get_dofs(inodes, self._get_solution_comps())
            coords = nodes.get_some_coords(inodes)

            eldisp = globdat[gn.STATE0][idofs]

            def exact_func(x):
                return pu.evaluate(str_eps_exact, [x], self._rank, extra_dict=eval_dict)

            def fem_func(x):
                sgrads = shape.eval_global_shape_gradients([x], coords)
                return eldisp @ sgrads

            def error_func(x):
                return (exact_func(x) - fem_func(x)) ** 2

            norm = np.sqrt(quad(error_func, coords[0, 0], coords[0, 1])[0])

            xtable.set_value(ielem, jcol, norm)

        table = xtable.to_table()
        return table

    def _get_solution_comps(self):
        if self._rank == 1:
            comps = ["dx"]
        elif self._rank == 2:
            comps = ["dx", "dy"]
        elif self._rank == 3:
            comps = ["dx", "dy", "dz"]
        return comps

    def _get_gradient_comps(self):
        if self._rank == 1:
            comps = ["xx"]
        elif self._rank == 2:
            comps = ["xx", "yy"]
        elif self._rank == 3:
            comps = ["xx", "yy", "zz"]
        return comps

    def _get_numpy_eval_dict(self):
        numpy_eval_dict = {
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "pi": np.pi,
        }

        return numpy_eval_dict

    def _get_sympy_eval_dict(self):
        sympy_eval_dict = {
            "x": sym.Symbol("x"),
            "y": sym.Symbol("y"),
            "z": sym.Symbol("z"),
            "exp": sym.exp,
            "sin": sym.sin,
            "cos": sym.cos,
            "tan": sym.tan,
            "sqrt": sym.sqrt,
            "pi": sym.pi,
        }

        return sympy_eval_dict

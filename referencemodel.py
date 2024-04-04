import sympy as sym

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util import to_xtable
from myjive.util.proputils import mdtarg, optarg


class ReferenceModel(Model):
    def GETEXACTTABLE(self, name, table, globdat, points, **kwargs):
        if "solution" in name:
            table = self._get_exact_solution(table, points)
        elif "strain" in name:
            table = self._get_exact_strain(table, points)
        elif "stress" in name:
            table = self._get_exact_stress(table, points)
        return table

    def configure(self, globdat, **props):
        # get props
        u = mdtarg(self, props, "u")
        kappa = mdtarg(self, props, "kappa")
        eval_params = optarg(self, props, "params", dtype=dict)

        # Get basic dimensionality info
        self._rank = globdat[gn.MESHRANK]
        self._strcount = self._rank * (self._rank + 1) // 2

        # Get the dictionary for load evaluation
        eval_dict = self._get_sympy_eval_dict()
        eval_dict.update(eval_params)

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

    def _get_gradient_comps(self):
        if self._rank == 1:
            comps = ["xx"]
        elif self._rank == 2:
            comps = ["xx", "yy"]
        elif self._rank == 3:
            comps = ["xx", "yy", "zz"]
        return comps

    def _get_sympy_eval_dict(self):
        sympy_eval_dict = {
            "x": sym.Symbol("x"),
            "y": sym.Symbol("y"),
            "z": sym.Symbol("z"),
            "exp": sym.exp,
            "sin": sym.sin,
            "cos": sym.cos,
            "tan": sym.tan,
            "pi": sym.pi,
        }

        return sympy_eval_dict

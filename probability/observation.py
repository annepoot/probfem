import numpy as np
from copy import deepcopy

from myjive.app import main
from myjivex.declare import declare_all
from myjive.names import GlobNames as gn
from myjive.util.proputils import (
    set_recursive,
    get_recursive,
    split_key,
    split_off_type,
)

__all__ = ["LinearObservationOperator", "FEMObservationOperator"]


class ObservationOperator:
    def calc_prediction(self, x):
        raise NotImplementedError("This has to be implemented in a child class")


class LinearObservationOperator(ObservationOperator):
    def __init__(self, *, operator):
        if not isinstance(operator, np.ndarray):
            raise TypeError
        if not len(operator.shape) == 2:
            raise ValueError

        self._operator = operator

    def calc_prediction(self, x):
        return self._operator @ x


class FEMObservationOperator(ObservationOperator):
    def __init__(
        self,
        forward_props,
        input_variables,
        output_type,
        output_variables,
        output_locations,
        output_dofs,
        run_modules,
    ):
        self._forward_props = deepcopy(forward_props)
        self._input_variables = input_variables
        self._out_type = output_type
        self._out_vars = output_variables
        self._out_locs = output_locations
        self._out_dofs = output_dofs

        if self._out_type not in ["nodal", "local"]:
            raise ValueError

        # run the full forward model at initialization
        self._globdat = main.jive(self._forward_props, extra_declares=[declare_all])
        self._run_modules = [self._globdat[gn.MODULES][name] for name in run_modules]

    def calc_prediction(self, x):
        if len(x) != len(self._input_variables):
            raise ValueError

        modelprops = self._forward_props[gn.MODEL]

        for xi, var in zip(x, self._input_variables):
            keys = split_key(var)
            assert get_recursive(modelprops, keys) is not None
            set_recursive(modelprops, keys, xi)

        for name, model in self._globdat[gn.MODELS].items():
            _, props = split_off_type(modelprops[name])
            model.configure(self._globdat, **props)

        for module in self._run_modules:
            module.run(self._globdat)

        output = np.zeros(len(self._out_vars))
        assert len(self._out_vars) == len(self._out_locs) == len(self._out_dofs)

        for i, (var, loc, dof) in enumerate(
            zip(self._out_vars, self._out_locs, self._out_dofs)
        ):
            if self._out_type == "nodal":
                pred = self._nodal_prediction(self._globdat, var, loc, dof)
            elif self._out_type == "local":
                pred = self._local_prediction(self._globdat, var, loc, dof)
            output[i] = pred

        return output

    def _nodal_prediction(self, globdat, var, loc, dof):
        keys = split_key(var)
        field = get_recursive(globdat, keys)
        idof = globdat[gn.DOFSPACE].get_dof(loc, dof)
        return field[idof]

    def _local_prediction(self, globdat, var, loc, dof):
        keys = split_key(var)
        field = get_recursive(globdat, keys)

        elems = globdat[gn.ESET]
        nodes = globdat[gn.NSET]
        dofspace = globdat[gn.DOFSPACE]
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss1")

        for inodes in elems:
            coords = nodes.get_some_coords(inodes)

            if shape.contains_global_point(loc, coords):
                idofs = dofspace.get_dofs(inodes, [dof])
                elfield = field[idofs]
                sfuncs = shape.eval_global_shape_functions(loc, coords)
                pred = sfuncs @ elfield
                break
        else:
            raise RuntimeError("No matching element found!")

        return pred

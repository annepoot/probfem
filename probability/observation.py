import numpy as np

from myjive.names import GlobNames as gn
from myjive.util.proputils import (
    set_recursive,
    get_recursive,
    split_key,
)

__all__ = [
    "LinearObservationOperator",
    "FEMObservationOperator",
    "RemeshFEMObservationOperator",
]


class ObservationOperator:
    def calc_prediction(self, x):
        raise NotImplementedError("This has to be implemented in a child class")


class LinearObservationOperator(ObservationOperator):
    def __init__(self, *, operator):
        if not isinstance(operator, np.ndarray):
            raise TypeError
        if not len(operator.shape) == 2:
            raise ValueError

        self.operator = operator

    def calc_prediction(self, x):
        return self.operator @ x


class FEMObservationOperator(ObservationOperator):
    def __init__(
        self,
        jive_runner,
        input_variables,
        output_type,
        output_variables,
        output_locations,
        output_dofs,
    ):
        self.jive_runner = jive_runner
        self.input_variables = input_variables
        self.output_type = output_type
        self.output_variables = output_variables
        self.output_locations = output_locations
        self.output_dofs = output_dofs

        if self.output_type not in ["nodal", "local"]:
            raise ValueError

    def calc_prediction(self, x):
        if len(x) != len(self.input_variables):
            raise ValueError

        for x_i, var in zip(x, self.input_variables):
            keys = split_key(var)
            assert get_recursive(self.jive_runner.props, keys) is not None
            set_recursive(self.jive_runner.props, keys, x_i)

        globdat = self.jive_runner()

        output = np.zeros(len(self.output_variables))
        assert (
            len(self.output_variables)
            == len(self.output_locations)
            == len(self.output_dofs)
        )

        for i, (var, loc, dof) in enumerate(
            zip(self.output_variables, self.output_locations, self.output_dofs)
        ):
            if self.output_type == "nodal":
                pred = self._nodal_prediction(globdat, var, loc, dof)
            elif self.output_type == "local":
                pred = self._local_prediction(globdat, var, loc, dof)
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
        shape = globdat[gn.SHAPE]

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


class RemeshFEMObservationOperator(ObservationOperator):
    def __init__(
        self,
        *,
        jive_runner,
        mesher,
        mesh_props,
        input_variables,
        output_locations,
        output_dofs,
    ):
        self.jive_runner = jive_runner
        self.mesher = mesher
        self.mesh_props = mesh_props
        self.input_variables = input_variables
        self.output_locations = output_locations
        self.output_dofs = output_dofs

    def calc_prediction(self, x):
        if len(x) != len(self.input_variables):
            raise ValueError

        for x_i, var in zip(x, self.input_variables):
            assert var in self.mesh_props
            self.mesh_props[var] = x_i

        nodes, elems = self.mesher(**self.mesh_props)[0]
        self.jive_runner.elems = elems
        globdat = self.jive_runner()

        output = np.zeros(len(self.output_locations))
        assert len(self.output_locations) == len(self.output_dofs)

        state0 = globdat["state0"]
        coords = globdat["nodeSet"].get_coords()
        dofs = globdat["dofSpace"]

        tol = 1e-8

        for i, (loc, dof) in enumerate(zip(self.output_locations, self.output_dofs)):
            inodes = np.where(np.all(abs(coords - loc) < tol, axis=1))[0]
            if len(inodes) == 0:
                output[i] = np.nan
            elif len(inodes) == 1:
                inode = inodes[0]
                idof = dofs.get_dof(inode, dof)
                output[i] = state0[idof]
            else:
                assert False

        return output

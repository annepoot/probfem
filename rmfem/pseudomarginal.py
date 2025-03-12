import numpy as np
from warnings import warn
from scipy.special import logsumexp

from myjive.fem.nodeset import to_xnodeset

from probability.observation import FEMObservationOperator, RemeshFEMObservationOperator
from probability.likelihood import Likelihood
from fem.meshing import (
    get_patches_around_nodes,
    calc_elem_sizes,
    calc_boundary_nodes,
    find_coords_in_nodeset,
)

from .perturbation import calc_perturbed_coords_cpp


__all__ = [
    "RMFEMObservationOperator",
    "RemeshRMFEMObservationOperator",
    "PseudoMarginalLikelihood",
]


class RMFEMObservationOperator(FEMObservationOperator):

    def __init__(self, *, p, seed, omit_coords=[], **kws):
        super().__init__(**kws)

        self.p = p
        self.rng = np.random.default_rng(seed)
        self.omit_coords = omit_coords

        self.elems = self.jive_runner.elems
        self.nodes = self.elems.get_nodes()
        self._patches = get_patches_around_nodes(self.elems)
        self._ref_coords = np.copy(self.nodes.get_coords())
        self._ref_elem_sizes = calc_elem_sizes(self.elems)
        self._boundary = calc_boundary_nodes(self.elems)
        self._omit_nodes = find_coords_in_nodeset(self.omit_coords, self.nodes)

    def calc_prediction(self, x):
        if not hasattr(self, "_perturbed") or not self._perturbed:
            warn("nodes have not been perturbed since the last prediction")

        self._perturbed = False

        return super().calc_prediction(x)

    def perturb_nodes(self):

        if hasattr(self, "_perturbed") and self._perturbed:
            warn("nodes have already been perturbed since the last prediction")

        new_coords = calc_perturbed_coords_cpp(
            ref_coords=self._ref_coords,
            elems=self.elems,
            elem_sizes=self._ref_elem_sizes,
            p=self.p,
            rng=self.rng,
            boundary=self._boundary,
            omit_nodes=self._omit_nodes,
            patches=self._patches,
        )

        to_xnodeset(self.nodes)
        self.nodes.set_coords(new_coords)
        self.nodes.to_nodeset()

        assert self.nodes == self.elems.get_nodes()

        self._perturbed = True


class RemeshRMFEMObservationOperator(RemeshFEMObservationOperator):

    def __init__(self, *, p, seed, omit_coords=[], **kws):
        super().__init__(**kws)

        self.p = p
        self.rng = np.random.default_rng(seed)
        self.omit_coords = omit_coords

    def restore_ref(self, x):
        for x_i, var in zip(x, self.input_variables):
            assert var in self.mesh_props
            self.mesh_props[var] = x_i

        self.nodes, self.elems = self.mesher(**self.mesh_props)

        self._patches = get_patches_around_nodes(self.elems)
        self._ref_coords = np.copy(self.nodes.get_coords())
        self._ref_elem_sizes = calc_elem_sizes(self.elems)
        self._boundary = calc_boundary_nodes(self.elems)

        if self.mandatory_coords is None:
            omit_nodes = find_coords_in_nodeset(self.omit_coords, self.nodes)
        else:
            omit_coords = np.concatenate([self.omit_coords, self.mandatory_coords])
            omit_nodes = find_coords_in_nodeset(omit_coords, self.nodes)

        if None in omit_nodes:
            self._invalid_mesh = True
        else:
            self._invalid_mesh = False
            self._omit_nodes = np.unique(omit_nodes)
            self._output_inodes = find_coords_in_nodeset(
                self.output_locations, self.nodes
            )

    def calc_prediction(self, x):
        if not hasattr(self, "_perturbed") or not self._perturbed:
            warn("nodes have not been perturbed since the last prediction")

        self._perturbed = False

        if self._invalid_mesh:
            return np.full(self.output_locations.shape[0], np.nan)

        if len(x) != len(self.input_variables):
            raise ValueError

        self.jive_runner.update_elems(self.elems)
        globdat = self.jive_runner("dofSpace", "state0")

        output = np.zeros(len(self.output_locations))
        assert len(self.output_locations) == len(self.output_dofs)

        state0 = globdat["state0"]
        dofs = globdat["dofSpace"]

        for i, (inode, dof) in enumerate(zip(self._output_inodes, self.output_dofs)):
            if inode is None:
                output[i] = np.nan
            else:
                idof = dofs.get_dof(inode, dof)
                output[i] = state0[idof]

        return output

    def perturb_nodes(self):
        if hasattr(self, "_perturbed") and self._perturbed:
            warn("nodes have already been perturbed since the last prediction")

        new_coords = calc_perturbed_coords_cpp(
            ref_coords=self._ref_coords,
            elems=self.elems,
            elem_sizes=self._ref_elem_sizes,
            p=self.p,
            rng=self.rng,
            boundary=self._boundary,
            omit_nodes=self._omit_nodes,
            patches=self._patches,
        )

        to_xnodeset(self.nodes)
        self.nodes.set_coords(new_coords)
        self.nodes.to_nodeset()

        assert self.nodes == self.elems.get_nodes()

        self._perturbed = True


class PseudoMarginalLikelihood(Likelihood):

    def __init__(self, likelihood, n_sample):
        assert isinstance(likelihood, Likelihood)
        assert isinstance(
            likelihood.operator,
            (RMFEMObservationOperator, RemeshRMFEMObservationOperator),
        )
        self.likelihood = likelihood
        self.n_sample = n_sample

    def calc_logpdf(self, x):
        logpdfs = np.zeros(self.n_sample)
        operator = self.likelihood.operator

        if isinstance(operator, RemeshRMFEMObservationOperator):
            operator.restore_ref(x)

        for i in range(self.n_sample):
            self.likelihood.operator.perturb_nodes()
            logpdfs[i] = self.likelihood.calc_logpdf(x)

        logmean = logsumexp(logpdfs) - np.log(len(logpdfs))
        return logmean

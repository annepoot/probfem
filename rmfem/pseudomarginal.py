import numpy as np
from warnings import warn
from scipy.special import logsumexp

from myjive.names import GlobNames as gn
from myjive.fem.nodeset import to_xnodeset

from probability.observation import FEMObservationOperator, RemeshFEMObservationOperator
from probability.likelihood import Likelihood
from fem.meshing import (
    get_patches_around_nodes,
    calc_elem_sizes,
    read_mesh,
    write_mesh,
    calc_boundary_nodes,
)

from .perturbation import calc_perturbed_coords, calc_perturbed_coords_cpp


__all__ = [
    "RMFEMObservationOperator",
    "RemeshRMFEMObservationOperator",
    "PseudoMarginalLikelihood",
]


class RMFEMObservationOperator(FEMObservationOperator):

    def __init__(self, *, p, seed, omit_nodes=[], **kws):
        super().__init__(**kws)

        self.p = p
        self.rng = np.random.default_rng(seed)
        self.omit_nodes = omit_nodes

        self.elems = self.globdat[gn.ESET]
        self.nodes = self.globdat[gn.NSET]
        self._patches = get_patches_around_nodes(self.elems)
        self._ref_coords = np.copy(self.nodes.get_coords())
        self._ref_elem_sizes = calc_elem_sizes(self.elems)
        self._boundary = calc_boundary_nodes(self.elems)

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
            omit_nodes=self.omit_nodes,
            patches=self._patches,
        )

        to_xnodeset(self.nodes)
        self.nodes.set_coords(new_coords)
        self.nodes.to_nodeset()
        assert self.nodes == self.globdat[gn.NSET]

        self._perturbed = True


class RemeshRMFEMObservationOperator(RemeshFEMObservationOperator):

    def __init__(self, *, p, seed, omit_nodes=[], **kws):
        super().__init__(**kws)

        self.p = p
        self.rng = np.random.default_rng(seed)
        self.omit_nodes = omit_nodes

    def restore_ref(self, x):
        for x_i, var in zip(x, self.input_variables):
            assert var in self.mesh_props
            self.mesh_props[var] = x_i

        self.mesher(**self.mesh_props)

        self.nodes, self.elems = read_mesh(self.mesh_props["fname"])
        self._patches = get_patches_around_nodes(self.elems)
        self._ref_coords = np.copy(self.nodes.get_coords())
        self._ref_elem_sizes = calc_elem_sizes(self.elems)
        self._boundary = calc_boundary_nodes(self.elems)

    def calc_prediction(self, x):
        if not hasattr(self, "_perturbed") or not self._perturbed:
            warn("nodes have not been perturbed since the last prediction")

        self._perturbed = False

        if len(x) != len(self.input_variables):
            raise ValueError

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
            omit_nodes=self.omit_nodes,
            patches=self._patches,
        )

        to_xnodeset(self.nodes)
        self.nodes.set_coords(new_coords)
        self.nodes.to_nodeset()

        assert self.nodes == self.elems.get_nodes()
        write_mesh(self.elems, self.mesh_props["fname"])

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

    def calc_pdf(self, x):
        pdfs = np.zeros(self.n_sample)
        operator = self.likelihood.operator

        if isinstance(operator, RemeshRMFEMObservationOperator):
            operator.restore_ref(x)

        for i in range(self.n_sample):
            self.likelihood.operator.perturb_nodes()
            pdfs[i] = self.likelihood.calc_pdf(x)

        return np.mean(pdfs)

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

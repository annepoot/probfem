from myjive.util.proputils import split_off_type
from scipy.special import logsumexp

from myjive.names import GlobNames as gn
from myjive.fem.nodeset import to_xnodeset

from probability.observation import FEMObservationOperator
from probability.likelihood import Likelihood
from fem.meshing import get_patch_around_node, calc_elem_sizes

import numpy as np
from warnings import warn

__all__ = ["PseudoMarginalLikelihood", "RMFEMObservationOperator"]


class RMFEMObservationOperator(FEMObservationOperator):

    def __init__(self, *, p, seed, **kws):
        super().__init__(**kws)

        self.p = p
        self.rng = np.random.default_rng(seed)

        self.elems = self._globdat[gn.ESET]
        self.nodes = self._globdat[gn.NSET]
        self._ref_coords = np.copy(self.nodes.get_coords())
        self._ref_elem_sizes = calc_elem_sizes(self.elems)

    def calc_prediction(self, x):
        if not hasattr(self, "_perturbed") or not self._perturbed:
            warn("nodes have not been perturbed since the last prediction")

        self._perturbed = False

        return super().calc_prediction(x)

    def perturb_nodes(self):

        if hasattr(self, "_perturbed") and self._perturbed:
            warn("nodes have already been perturbed since the last prediction")

        new_coords = self._calc_perturbed_coords(
            self._ref_coords, self.elems, self._ref_elem_sizes
        )

        to_xnodeset(self.nodes)
        self.nodes.set_coords(new_coords)
        self.nodes.to_nodeset()
        assert self.nodes == self._globdat[gn.NSET]

        self._perturbed = True

    def _calc_perturbed_coords(self, ref_coords, elems, elem_sizes):
        h = np.max(elem_sizes)
        rank = ref_coords.shape[1]
        pert_coords = np.copy(ref_coords)

        for inode, coords in enumerate(pert_coords):
            # if inode in self._omit_nodes:
            #     continue

            if rank == 1:
                alpha_i_bar = np.array([self.rng.uniform(-0.5, 0.5)])
            elif rank == 2:
                r = 0.25 * np.sqrt(self.rng.uniform(0.0, 1.0))
                theta = self.rng.uniform(0.0, 2 * np.pi)
                alpha_i_bar = r * np.array([np.cos(theta), np.sin(theta)])
            else:
                raise NotImplementedError(
                    "RandomMeshModel has not been implemented for 3D yet"
                )

            patch = get_patch_around_node(inode, elems)
            h_i_bar = np.min(elem_sizes[patch])
            alpha_i = (h_i_bar / h) ** self.p * alpha_i_bar

            warn("boundary conditions have not been implemented properly")
            # for group, dof in zip(self._bgroups, self._bdofs):
            #     ngroup = globdat[gn.NGROUPS][group]
            #     if inode in ngroup:
            #         if dof == "dx":
            #             alpha_i[0] = 0.0
            #         elif dof == "dy":
            #             alpha_i[1] = 0.0
            #         elif dof == "dz":
            #             alpha_i[2] = 0.0
            if inode in [0, ref_coords.shape[0] - 1]:
                alpha_i *= 0.0

            coords += h**self.p * alpha_i

            pert_coords[inode] = coords

        return pert_coords


class PseudoMarginalLikelihood(Likelihood):

    def __init__(self, likelihood, n_sample):
        likelihood_cls, likelihood_kws = split_off_type(likelihood)

        assert issubclass(likelihood_cls, Likelihood)

        self.likelihood = likelihood_cls(**likelihood_kws)
        self.n_sample = n_sample

    def calc_pdf(self, x):
        pdfs = np.zeros(self.n_sample)

        for i in range(self.n_sample):
            self.likelihood.operator.perturb_nodes()
            pdfs[i] = self.likelihood.calc_pdf(x)

        return np.mean(pdfs)

    def calc_logpdf(self, x):
        logpdfs = np.zeros(self.n_sample)

        for i in range(self.n_sample):
            self.likelihood.operator.perturb_nodes()
            logpdfs[i] = self.likelihood.calc_logpdf(x)

        logmean = logsumexp(logpdfs) - np.log(len(logpdfs))
        return logmean

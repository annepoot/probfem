from copy import deepcopy

from bayes.gaussian import DirectGaussian, LinTransGaussian, LinSolveGaussian
from myjive.names import GlobNames as gn
from myjive.model import Model
from myjive.solver import Constrainer
from myjive.util.proputils import split_off_type


class BFEMModel(Model):
    def ASKMATRICES(self):
        return self._ask_matrices()

    def RETURNMATRICES(self, globdat):
        return self._return_matrices(globdat)

    def GETPRIOR(self, globdat):
        prior = self._get_prior(globdat)
        return prior

    def APPLYPOSTTRANS(self, distribution, globdat):
        return self._apply_post_trans(distribution, globdat)

    @Model.save_config
    def configure(self, globdat, *, prior, postTrans):

        self._priorprops = prior
        self._transprops = postTrans

        self._fullpriorprops = deepcopy(self._priorprops)
        self._fulltransprops = deepcopy(self._transprops)

    def _ask_matrices(self):
        needed = []
        needed = self._recurse_askmat(self._priorprops, needed)
        needed = self._recurse_askmat(self._transprops, needed)
        return needed

    def _recurse_askmat(self, props, needed):
        dic = {
            "DirectGaussian": ["cov"],
            "LinTransGaussian": ["scale"],
            "LinSolveGaussian": ["inv"],
            "Prior": [],
        }

        field_list = dic[props["type"]]

        for field in field_list:
            if props[field] in ["K", "M"]:
                needed.append(props[field])

        if "latent" in props:
            needed = self._recurse_askmat(props["latent"], needed)

        return needed

    def _return_matrices(self, globdat):
        self._fullpriorprops = self._recurse_retmat(globdat, self._fullpriorprops)
        self._fulltransprops = self._recurse_retmat(globdat, self._fulltransprops)

    def _recurse_retmat(self, globdat, props, latent=None):
        dic = {
            "DirectGaussian": ["cov"],
            "LinTransGaussian": ["scale"],
            "LinSolveGaussian": ["inv"],
            "Prior": [],
        }

        field_list = dic[props["type"]]

        for field in field_list:
            if props[field] == "K":
                K = globdat[gn.MATRIX0]
                c = globdat[gn.CONSTRAINTS]
                conman = Constrainer(c, K)
                props[field] = conman.get_output_matrix().toarray()
            elif props[field] == "M":
                M = globdat[gn.MATRIX2]
                c = globdat[gn.CONSTRAINTS]
                conman = Constrainer(c, M)
                props[field] = conman.get_output_matrix().toarray()

        if "latent" in props:
            props["latent"] = self._recurse_retmat(globdat, props["latent"])

        return props

    def _get_prior(self, globdat):
        prior = self._recurse_prior(globdat, self._fullpriorprops)
        return prior

    def _recurse_prior(self, globdat, fullprops):
        disttype, distprops = split_off_type(fullprops)

        if "latent" in distprops:
            latent = self._recurse_prior(globdat, distprops["latent"])
            distprops.pop("latent")

        if disttype == "DirectGaussian":
            return DirectGaussian(**distprops)
        elif disttype == "LinTransGaussian":
            return LinTransGaussian(latent, **distprops)
        elif disttype == "LinSolveGaussian":
            return LinSolveGaussian(latent, **distprops)

    def _apply_post_trans(self, distribution, globdat):
        transformed = self._recurse_trans(globdat, self._fulltransprops, distribution)
        return transformed

    def _recurse_trans(self, globdat, fullprops, prior):
        disttype, distprops = split_off_type(fullprops)

        if "latent" in distprops:
            if distprops["latent"]["type"] != "Prior":
                latent = self._recurse_trans(globdat, distprops["latent"], prior)
            else:
                latent = prior

            distprops.pop("latent")

        if disttype == "DirectGaussian":
            return DirectGaussian(**distprops)
        elif disttype == "LinTransGaussian":
            return LinTransGaussian(latent, **distprops)
        elif disttype == "LinSolveGaussian":
            return LinSolveGaussian(latent, **distprops)

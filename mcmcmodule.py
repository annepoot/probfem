import numpy as np
from scipy.stats import multivariate_normal, uniform
from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import (
    check_dict,
    check_list,
    split_off_type,
    set_recursive,
    get_recursive,
)


class MCMCModule(Module):
    @Module.save_config
    def configure(
        self,
        globdat,
        *,
        solveModule,
        nsample,
        variables,
        startValue=None,
        priorMean=None,
        priorStd,
        proposalMean=None,
        proposalStd,
        corruptionNoise,
        observationNoise,
        seed=None,
        output=["variables"]
    ):
        # Validate input arguments
        check_dict(self, solveModule, ["type"])
        check_list(self, output)
        self._nsample = nsample
        self._variables = variables
        self._nvar = len(self._variables)
        self._start = np.zeros(self._nvar) if startValue is None else startValue
        self._mu_0 = np.zeros(self._nvar) if priorMean is None else priorMean
        self._sigma_0 = priorStd
        self._sigma_q = proposalStd
        self._sigma_c = corruptionNoise
        self._sigma_y = observationNoise
        self._rng = np.random.default_rng(seed)
        self._output = output

        modulefac = globdat[gn.MODULEFACTORY]
        solvetype, solveprops = split_off_type(solveModule)
        self._solvemodule = modulefac.get_module(solvetype, "solveModule")
        self._solvemodule.configure(globdat, **solveprops)

    def init(self, globdat):
        # Initialize solvemodule
        self._solvemodule.init(globdat)

    def run(self, globdat):
        models = globdat[gn.MODELS]
        observation_model = self.get_unique_relevant_model("GETLOGLIKELIHOOD", models)

        self._solvemodule.run(globdat)

        # Initial guess
        xi = self._start
        self._configure_models(globdat, self._variables, xi)
        self._solvemodule.run(globdat)
        solution = globdat[gn.STATE0]
        stiffness = globdat[gn.TABLES]["stiffness"][""]

        Sigma_0 = self._sigma_0**2 * np.identity(self._nvar)
        prior = multivariate_normal(mean=self._mu_0, cov=Sigma_0)
        Sigma_q = self._sigma_q**2 * np.identity(self._nvar)
        proposal = multivariate_normal(mean=xi, cov=Sigma_q)

        output_dict = {}
        output_dict = self._update_output_dict(output_dict, globdat, 0, xi, True)

        logprior = prior.logpdf(xi)
        loglikelihood = observation_model.GETLOGLIKELIHOOD(globdat)

        for i in range(1, self._nsample + 1):
            proposal.mean = xi
            xi_prop = xi + self._sigma_q * self._rng.standard_normal(4)

            self._configure_models(globdat, self._variables, xi_prop)
            self._solvemodule.run(globdat)

            logprior_prop = prior.logpdf(xi_prop)
            loglikelihood_prop = observation_model.GETLOGLIKELIHOOD(globdat)

            logalpha = logprior_prop + loglikelihood_prop - logprior - loglikelihood

            if logalpha < 0:
                if self._rng.uniform() < np.exp(logalpha):
                    accept = True
                else:
                    accept = False
            else:
                accept = True

            if accept:
                xi = xi_prop
                logprior = logprior_prop
                loglikelihood = loglikelihood_prop

            output_dict = self._update_output_dict(output_dict, globdat, i, xi, accept)

        globdat["mcmc"] = output_dict

        return "ok"

    def shutdown(self, globdat):
        pass

    def _configure_models(self, globdat, variables, values):
        for var, val in zip(self._variables, values):
            name, keys = var.split(".")[0], var.split(".")[1:]
            for model in globdat[gn.MODELS]:
                if model.get_name() == name:
                    config = model.get_config()
                    set_recursive(config, keys, val)
                    _, config = split_off_type(config)
                    model.configure(globdat, **config)

    def _update_output_dict(self, output_dict, globdat, i, xi, accept):
        if accept:
            for key in self._output:
                if key == "variables":
                    value = xi
                else:
                    if "." in key:
                        value = get_recursive(globdat, key.split("."))
                    else:
                        value = globdat[key]

                if i == 0:
                    output_dict[key] = np.zeros((self._nsample + 1, len(value)))
                output_dict[key][i] = value
        else:
            for key, value in output_dict.items():
                output_dict[key][i] = output_dict[key][i - 1]

        return output_dict

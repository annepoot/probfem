import numpy as np
from scipy.stats import multivariate_normal, uniform
from myjive.names import GlobNames as gn
from myjive.app import Module
from myjive.util.proputils import check_dict, split_off_type


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
        seed=None
    ):
        # Validate input arguments
        check_dict(self, solveModule, ["type"])
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

        modulefac = globdat[gn.MODULEFACTORY]
        solvetype, solveprops = split_off_type(solveModule)
        self._solvemodule = modulefac.get_module(solvetype, "solveModule")
        self._solvemodule.configure(globdat, **solveprops)

    def init(self, globdat):
        # Initialize solvemodule
        self._solvemodule.init(globdat)

    def run(self, globdat):
        self._solvemodule.run(globdat)
        ref_solution = globdat[gn.STATE0]

        ndof = globdat[gn.DOFSPACE].dof_count()
        measure_dofs = np.arange(0, ndof)[:: (ndof - 1) // 10][1:-1]
        # ref = ref_solution[measure_dofs]
        ref = np.array(
            [
                0.01154101,
                0.01667733,
                0.01592942,
                0.00980423,
                -0.00043005,
                -0.01177105,
                -0.02001336,
                -0.0211289,
                -0.01350695,
            ]
        )

        measurements = ref + self._sigma_c * self._rng.standard_normal(9)

        # Initial guess
        xi = self._start
        self._configure_models(globdat, self._variables, xi)
        self._solvemodule.run(globdat)
        solution = globdat[gn.STATE0]
        stiffness = globdat[gn.TABLES]["stiffness"][""]

        mu_y = globdat["state0"][measure_dofs]
        Sigma_0 = self._sigma_0**2 * np.identity(self._nvar)
        prior = multivariate_normal(mean=self._mu_0, cov=Sigma_0)
        Sigma_q = self._sigma_q**2 * np.identity(self._nvar)
        proposal = multivariate_normal(mean=xi, cov=Sigma_q)
        Sigma_y = self._sigma_y**2 * np.identity(len(mu_y))

        variables = np.zeros((self._nsample + 1, len(xi)))
        solutions = np.zeros((self._nsample + 1, len(solution)))
        stiffnesses = np.zeros((self._nsample + 1, len(solution)))
        variables[0] = xi
        solutions[0] = solution
        stiffnesses[0] = stiffness

        logp_prior = prior.logpdf(xi)
        logp_likelihood = multivariate_normal(solution[measure_dofs], Sigma_y).logpdf(
            measurements
        )

        for i in range(1, self._nsample + 1):
            proposal.mean = xi
            xi_prop = xi + self._sigma_q * self._rng.standard_normal(4)

            self._configure_models(globdat, self._variables, xi_prop)
            self._solvemodule.run(globdat)

            solution_prop = globdat[gn.STATE0]
            stiffness_prop = globdat[gn.TABLES]["stiffness"][""]

            logp_prior_prop = prior.logpdf(xi_prop)
            logp_likelihood_prop = multivariate_normal(
                solution_prop[measure_dofs], Sigma_y
            ).logpdf(measurements)

            logalpha = min(
                logp_prior_prop + logp_likelihood_prop - logp_prior - logp_likelihood, 0
            )

            if self._rng.uniform() < np.exp(logalpha):
                xi = xi_prop
                solution = solution_prop
                stiffness = stiffness_prop
                logp_prior = logp_prior_prop
                logp_likelihood = logp_likelihood_prop

            print(xi)

            variables[i] = xi
            stiffnesses[i] = stiffness
            solutions[i] = solution

        globdat["mcmcvariables"] = variables
        globdat["mcmcsolutions"] = solutions
        globdat["mcmcstiffnesses"] = stiffnesses

        return "ok"

    def shutdown(self, globdat):
        pass

    def _configure_models(self, globdat, variables, values):
        for var, val in zip(self._variables, values):
            name, keys = var.split(".")[0], var.split(".")[1:]
            for model in globdat[gn.MODELS]:
                if model.get_name() == name:
                    config = model.get_config()
                    subconf = config
                    for i, key in enumerate(keys):
                        if i == len(keys) - 1:
                            subconf[key] = val
                        else:
                            subconf = subconf[key]
                    _, config = split_off_type(config)
                    model.configure(globdat, **config)

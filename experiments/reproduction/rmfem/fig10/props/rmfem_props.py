from probability.sampling import RandomWalkMetropolisSampler
from .mcmc_props import mcmc_props


mwmc_props = {
    "inner": {"type": RandomWalkMetropolisSampler, **mcmc_props},
    "p": 1,
    "n_sample": 50,
    "seed": 0,
    "update_type": "modify_file",
    "fname": "1d-lin.mesh",
    "globdat_keys": ["state0", "elemSet", "nodeSet"],
}

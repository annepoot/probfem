from myjive.app import main
from sampling import MCMCRunner
from mcmc_props import mcmc_props
from fem_props import fem_props


rmfem_props = {
    "inner": {"type": main.jive, **fem_props},
    "p": 1,
    "n_sample": 50,
    "seed": 0,
    "update_type": "in_place",
    "globdat_keys": ["state0", "elemSet", "nodeSet"],
}

mwmc_props = {
    "inner": {"type": MCMCRunner, **mcmc_props},
    "p": 1,
    "n_sample": 50,
    "seed": 0,
    "update_type": "modify_file",
    "fname": "1d-lin.mesh",
    "globdat_keys": ["state0", "elemSet", "nodeSet"],
}

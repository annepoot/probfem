from rmfem import PseudoMarginalLikelihood, RMFEMObservationOperator
from .rwm_fem_props import get_rwm_fem_props


def get_rwm_rmfem_props(*, std_corruption, sigma_e, n_rep_obs, n_pseudomarginal):
    rwm_rmfem_props = get_rwm_fem_props(
        std_corruption=std_corruption, sigma_e=sigma_e, n_rep_obs=n_rep_obs
    )

    pmlikelihood_props = {
        "type": PseudoMarginalLikelihood,
        "likelihood": rwm_rmfem_props["target"].pop("likelihood"),
        "n_sample": n_pseudomarginal,
    }
    rwm_rmfem_props["target"]["likelihood"] = pmlikelihood_props

    obsoperator_props = pmlikelihood_props["likelihood"]["operator"]
    obsoperator_props["type"] = RMFEMObservationOperator
    obsoperator_props["p"] = 1
    obsoperator_props["seed"] = 0

    rwm_rmfem_props["recompute_logpdf"] = True

    return rwm_rmfem_props

from fem_props import get_fem_props
from probability.process import (
    ProjectedPrior,
    GaussianProcess,
    InverseCovarianceOperator,
    NaturalCovarianceOperator,
)


def get_bfem_props(covariance):
    coarse_fem_props = get_fem_props("meshes/plate_r0.msh")
    fine_fem_props = get_fem_props("meshes/plate_r1.msh")

    if covariance == "K":
        cov = InverseCovarianceOperator(fine_fem_props["model"])
    elif covariance == "M":
        cov = NaturalCovarianceOperator(
            fine_fem_props["model"], lumped_mass_matrix=False
        )
    else:
        assert False

    bfem_props = {
        "prior": {
            "type": ProjectedPrior,
            "prior": {
                "type": GaussianProcess,
                "mean": None,
                "cov": cov,
            },
            "init_props": fine_fem_props["init"],
            "solve_props": fine_fem_props["solve"],
        },
        "init_props": coarse_fem_props["init"],
        "solve_props": coarse_fem_props["solve"],
    }
    return bfem_props

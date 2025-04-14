import os
import numpy as np
import pytest

from fem.meshing import mesh_interval_with_line2
from probability.sampling import MCMCRunner
from probability.multivariate import Gaussian


def linear_tempering(i):
    if i > n_burn:
        return 1.0
    else:
        return i / n_burn


n_burn = 10000
n_sample = 100
tempering = linear_tempering


@pytest.mark.inverse
@pytest.mark.reproduction
@pytest.mark.values
def test_pullout_reproduction_values():
    from experiments.reproduction.inverse.pullout_bar.props import (
        get_rwm_fem_target,
        get_rwm_bfem_target,
        get_rwm_rmfem_target,
        get_rwm_statfem_target,
    )

    ref_map = get_pullout_reference_value_map()

    std_corruption = 1e-3
    n_elem_range = [16]

    for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

        if fem_type == "fem":
            sigma_e = std_corruption
            recompute_logpdf = False

        elif fem_type == "bfem":
            scale = "mle"  # f_c.T @ u_c / n_c
            sigma_e = std_corruption
            recompute_logpdf = False

        elif fem_type == "rmfem":
            sigma_e = std_corruption
            n_pseudomarginal = 10
            recompute_logpdf = True

        elif fem_type == "statfem":
            sigma_e = std_corruption
            recompute_logpdf = False

        for i, n_elem in enumerate(n_elem_range):
            nodes, elems = mesh_interval_with_line2(n=n_elem)

            if fem_type == "fem":
                target = get_rwm_fem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                )
            elif fem_type == "bfem":
                ref_nodes, ref_elems = mesh_interval_with_line2(n=2 * n_elem)
                target = get_rwm_bfem_target(
                    obs_elems=elems,
                    ref_elems=ref_elems,
                    std_corruption=std_corruption,
                    scale=scale,  # f_c.T @ u_c / n_c
                    sigma_e=sigma_e,
                )
            elif fem_type == "rmfem":
                target = get_rwm_rmfem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    n_pseudomarginal=n_pseudomarginal,
                    omit_nodes=False,
                )
            elif fem_type == "statfem":
                target = get_rwm_statfem_target(
                    elems=elems,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                )
            else:
                raise ValueError

            start_value = target.prior.calc_mean()
            proposal = Gaussian(start_value, target.prior.calc_cov())
            mcmc = MCMCRunner(
                target=target,
                proposal=proposal,
                n_sample=n_sample,
                n_burn=n_burn,
                start_value=start_value,
                seed=0,
                tempering=tempering,
                recompute_logpdf=recompute_logpdf,
                return_info=True,
            )
            samples, info = mcmc()

            last_sample = samples[-1]
            last_logpdf = info["loglikelihood"][-1]

            ref_sample = ref_map[fem_type][n_elem][:-1]
            ref_logpdf = ref_map[fem_type][n_elem][-1]

            assert np.allclose(last_sample, ref_sample)
            assert np.isclose(last_logpdf, ref_logpdf)


@pytest.mark.inverse
@pytest.mark.reproduction
@pytest.mark.values
def test_three_point_reproduction_values():
    from experiments.reproduction.inverse.three_point_hole.props import (
        get_rwm_fem_target,
        get_rwm_bfem_target,
        get_rwm_rmfem_target,
        get_rwm_statfem_target,
    )

    ref_map = get_three_point_reference_value_map()

    std_corruption = 1e-4
    h_range = [0.2]
    h_meas = 0.5

    folder = os.path.join("experiments", "reproduction", "inverse", "three_point_hole")

    for fem_type in ["fem", "bfem", "rmfem", "statfem"]:

        if fem_type == "fem":
            sigma_e = std_corruption
            recompute_logpdf = False

        elif fem_type == "bfem":
            sigma_e = std_corruption
            recompute_logpdf = False

        elif fem_type == "rmfem":
            sigma_e = std_corruption
            n_pseudomarginal = 10
            recompute_logpdf = True

        elif fem_type == "statfem":
            sigma_e = std_corruption
            recompute_logpdf = False

        for h in h_range:
            if fem_type == "fem":
                target = get_rwm_fem_target(
                    h=h,
                    h_meas=h_meas,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    folder=folder,
                )
            elif fem_type == "bfem":
                target = get_rwm_bfem_target(
                    h=h,
                    h_meas=h_meas,
                    std_corruption=std_corruption,
                    scale="mle",  # f_c.T @ u_c / n_c
                    sigma_e=sigma_e,
                    folder=folder,
                )
            elif fem_type == "rmfem":
                target = get_rwm_rmfem_target(
                    h=h,
                    h_meas=h_meas,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    n_pseudomarginal=n_pseudomarginal,
                    folder=folder,
                )
            elif fem_type == "statfem":
                target = get_rwm_statfem_target(
                    h=h,
                    h_meas=h_meas,
                    std_corruption=std_corruption,
                    sigma_e=sigma_e,
                    folder=folder,
                )
            else:
                raise ValueError

            start_value = target.prior.latent.calc_mean()
            proposal = Gaussian(start_value, target.prior.latent.calc_cov())

            mcmc = MCMCRunner(
                target=target,
                proposal=proposal,
                n_sample=n_sample,
                n_burn=n_burn,
                start_value=start_value,
                seed=np.random.default_rng(0),
                tempering=tempering,
                recompute_logpdf=recompute_logpdf,
                return_info=True,
            )

            samples, info = mcmc()

            last_sample = samples[-1]
            last_logpdf = info["loglikelihood"][-1]

            ref_sample = ref_map[fem_type][h][:-1]
            ref_logpdf = ref_map[fem_type][h][-1]

            assert np.allclose(last_sample, ref_sample)
            assert np.isclose(last_logpdf, ref_logpdf)


def get_pullout_reference_value_map():
    ref_map = {
        "fem": {
            16: [
                -0.26996043678474796,
                4.2781344056788795,
                -6.556288188826591,
            ],
        },
        "bfem": {
            16: [
                -0.08311372390643845,
                4.575932717320207,
                1.8624954298419891,
            ],
        },
        "rmfem": {
            16: [
                -0.24084434733610258,
                4.249841230491143,
                -6.482452843444126,
            ],
        },
        "statfem": {
            16: [
                0.12738075819738873,
                4.496511196479788,
                0.006980278124023576,
                3.8695027848289634,
                -3.610085720762406,
                -5.9445461015036125,
            ],
        },
    }

    return ref_map


def get_three_point_reference_value_map():
    ref_map = {
        "fem": {
            0.2: [
                0.322070592807296,
                0.23572240353162865,
                0.023570092705027995,
                5.633140894213423,
                0.29182919808693497,
                -8.03600831341388,
            ],
        },
        "bfem": {
            0.2: [
                1.3760729633513762,
                0.7143996318776058,
                0.3720924392454529,
                2.746539360291802,
                0.45336007811911205,
                1.3175281602137607,
            ],
        },
        "rmfem": {
            0.2: [
                1.205026591944789,
                0.686392929006952,
                0.32548253923277404,
                3.468428019800476,
                0.4052418517758896,
                -5.66616845893032,
            ],
        },
        "statfem": {
            0.2: [
                0.6024049648212163,
                0.39181878914725066,
                0.1905021398442981,
                2.749843271367154,
                0.38507145948254784,
                -0.10763832752763944,
                -0.3251389668377866,
                -6.884474456818794,
                -2.645035036926936,
            ],
        },
    }

    return ref_map

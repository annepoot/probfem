import numpy as np
import matplotlib.pyplot as plt

from fem.jive import CJiveRunner
from probability.multivariate import Gaussian
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params


def calc_snapshots(*, n_snapshot, h):

    #########################
    # get precomputed stuff #
    #########################

    nodes, elems, egroups = caching.get_or_calc_mesh(h=h)
    egroup = egroups["matrix"]
    ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
    distances = caching.get_or_calc_distances(egroup=egroup, h=h)

    backdoor = {}
    backdoor["xcoord"] = ipoints[:, 0]
    backdoor["ycoord"] = ipoints[:, 1]
    backdoor["e"] = np.zeros(ipoints.shape[0])

    snapshots = np.zeros((n_snapshot, 2 * len(nodes)))

    domain = np.linspace(0.0, 0.2, 101)

    inf_prior = GaussianProcess(
        mean=ZeroMeanFunction(),
        cov=SquaredExponential(l=0.02, sigma=2.0),
    )

    std_pd = 1e-6
    cov = inf_prior.calc_cov(domain, domain) + std_pd**2 * np.identity(len(domain))
    prior = Gaussian(mean=None, cov=cov, use_scipy_latent=False)

    rng = np.random.default_rng(0)

    x_coords = nodes.get_coords()[:, 0]
    load_inodes = np.where(np.logical_and(x_coords > -1.0, x_coords < 1.0))[0]
    rng.shuffle(load_inodes)
    coords = rng.uniform(-1.0, 1.0, size=(n_snapshot, 2))
    thetas = rng.uniform(0, 2 * np.pi, size=n_snapshot)

    l = 0.2
    A = 1 / (2 * np.pi * l**2)

    idx_mapper = (len(domain) - 1) / np.max(domain)

    xload = "cos({theta}) * {A} * exp(-0.5 * ((x - {x0})^2 + (y - {y0})^2) / {l}^2)"
    yload = "sin({theta}) * {A} * exp(-0.5 * ((x - {x0})^2 + (y - {y0})^2) / {l}^2)"

    for i in range(n_snapshot):
        if i % 10 == 0:
            print("snapshot", i)

        sample = prior.calc_sample(rng)
        damage = misc.sigmoid(sample, 1.0, 0.0)
        E_matrix = params.material_params["E_matrix"]

        for ip, ipoint in enumerate(ipoints):
            dist = distances[ip]
            idx_l = int(dist * idx_mapper)
            idx_r = idx_l + 1

            x_l = domain[idx_l]
            x_r = domain[idx_r]
            d_l = damage[idx_l]
            d_r = damage[idx_r]

            assert x_l <= dist <= x_r

            dam = d_l + (dist - x_l) / (x_r - x_l) * (d_r - d_l)
            backdoor["e"][ip] = E_matrix * (1 - dam)

        elem_stiffness = np.zeros(len(elems))

        for group_name, egroup in egroups.items():
            if group_name == "matrix":
                for ie, ielem in enumerate(egroup):
                    ip_stiffness = backdoor["e"][3 * ie : 3 * (ie + 1)]
                    elem_stiffness[ielem] = np.mean(ip_stiffness)
            elif group_name == "fiber":
                ielems = egroup.get_indices()
                elem_stiffness[ielems] = 0
            else:
                assert False

        props = get_fem_props()

        theta = thetas[i]
        x0 = coords[i, 0]
        y0 = coords[i, 0]

        props["model"]["model"]["models"].append("load")
        props["model"]["model"]["load"] = {
            "type": "Load",
            "elements": "all",
            "load": [
                xload.format(A=A, theta=theta, x0=x0, y0=y0, l=l),
                yload.format(A=A, theta=theta, x0=x0, y0=y0, l=l),
            ],
            "dofs": ["dx", "dy"],
        }

        jive = CJiveRunner(props, elems=elems, egroups=egroups)
        globdat = jive(**backdoor)

        snapshots[i] = globdat["state0"]

    return snapshots


def calc_lifting(*, h):
    nodes, elems, egroups = caching.get_or_calc_mesh(h=h)
    egroup = egroups["matrix"]
    ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
    ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=egroup, h=h)

    backdoor = {}
    backdoor["xcoord"] = ipoints[:, 0]
    backdoor["ycoord"] = ipoints[:, 1]
    backdoor["e"] = ip_stiffnesses

    props = get_fem_props()
    jive = CJiveRunner(props, elems=elems, egroups=egroups)
    globdat = jive(**backdoor)

    dofs = globdat["dofSpace"]
    xcoords = globdat["nodeSet"].get_coords()[:, 0]
    xdofs = dofs.get_dofs(range(len(nodes)), ["dx"])

    umax = -0.01
    xmin = -1.0
    xmax = 1.0

    lifting = np.zeros_like(globdat["state0"])
    lifting[xdofs] = (xcoords - xmin) / (xmax - xmin) * umax
    return lifting

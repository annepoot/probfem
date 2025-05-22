import numpy as np
import matplotlib.pyplot as plt

from fem.jive import CJiveRunner
from probability.multivariate import Gaussian
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]
h = 0.05

x = np.linspace(0.0, 0.2, 101)

target = GaussianProcess(
    mean=ZeroMeanFunction(),
    cov=SquaredExponential(l=0.02, sigma=2.0),
)

std_pd = 1e-6
fixed_cov = target.calc_cov(x, x) + std_pd**2 * np.identity(len(x))
fixed_target = Gaussian(mean=None, cov=fixed_cov, use_scipy_latent=False)

rng = np.random.default_rng(0)

E_matrix = params.material_params["E_matrix"]
alpha = params.material_params["alpha"]
beta = params.material_params["beta"]
c = params.material_params["c"]
d = params.material_params["d"]

saturation = misc.saturation(x, alpha, beta, c)
damage = misc.damage(saturation, d)
E = E_matrix * (1 - damage)

fig, ax = plt.subplots()

damage_samples = []

for i in range(20):
    sample = fixed_target.calc_sample(rng)
    damage_sample = misc.sigmoid(sample, 1.0, 0.0)
    damage_samples.append(damage_sample)
    ax.plot(x, damage_sample, color="C0", alpha=0.5)

damage_samples = np.array(damage_samples)
ax.plot(x, damage, color="k")
ax.set_title("Prior samples of damage field")
ax.set_xlim((0, 0.2))
ax.set_ylim((-0.1, 1.1))
ax.set_xlabel(r"Distance to fiber")
ax.set_ylabel(r"Damage")
ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
ax.set_yticks([0.0, 0.5, 1.0])
plt.show()

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

from myjivex.util import ElemViewer

for damage in damage_samples:
    for ip, ipoint in enumerate(ipoints):
        dist = distances[ip]
        idx_l = int(dist / np.max(x) * (len(x) - 1))
        idx_r = idx_l + 1

        x_l = x[idx_l]
        x_r = x[idx_r]
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
    jive = CJiveRunner(props, elems=elems, egroups=egroups)
    globdat = jive(**backdoor)

    ElemViewer(
        elem_stiffness,
        globdat,
        maxcolor=E_matrix,
        title=r"stiffness, $N_e = {}$".format(len(elems)),
    )

    # eps_xx, eps_yy, gamma_xy = misc.calc_strains(globdat)

    # ElemViewer(
    #     eps_xx[:, 0],
    #     globdat,
    #     maxcolor=0.01,
    #     title=r"max strain, $N_e = {}$".format(len(elems)),
    # )

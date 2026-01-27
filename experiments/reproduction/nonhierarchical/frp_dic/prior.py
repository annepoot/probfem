import numpy as np
from scipy.sparse import diags_array
import matplotlib as mpl
import matplotlib.pyplot as plt

from myjivex.util import ElemViewer

from fem.jive import CJiveRunner
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from util.linalg import Matrix

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

rve_size = params.geometry_params["rve_size"]
n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
tol = params.geometry_params["tol_fiber"]
seed = params.geometry_params["seed_fiber"]
h = 0.02

x = np.linspace(0.0, 0.2, 101)

target = GaussianProcess(
    mean=ZeroMeanFunction(),
    cov=SquaredExponential(l=0.02, sigma=2.0),
)

U, s, _ = np.linalg.svd(target.calc_cov(x, x))

trunc = 10
eigenfuncs = U[:, :trunc]
eigenvalues = s[:trunc]

kl_cov = SymbolicCovariance(Matrix(diags_array(eigenvalues), name="S"))
kl_target = Gaussian(mean=None, cov=kl_cov)

rng = np.random.default_rng(0)

E_matrix = params.material_params["E_matrix"]
alpha = params.material_params["alpha"]
beta = params.material_params["beta"]
c = params.material_params["c"]
d = params.material_params["d"]

saturation = misc.saturation(x, alpha, beta, c)
true_damage = misc.damage(saturation, d) * 100
E = E_matrix * (1 - true_damage)

damage_samples = []

for i in range(20):
    sample = eigenfuncs @ kl_target.calc_sample(rng)
    damage_sample = misc.sigmoid(sample, 1.0, 0.0) * 100
    damage_samples.append(damage_sample)

damage_samples = np.array(damage_samples)

color = mpl.colormaps["viridis"](0.5)
opacity = 0.3

fig, ax = plt.subplots()
for damage_sample in damage_samples:
    ax.plot(x, damage_sample, color=color, alpha=opacity)
ax.plot(x, true_damage, color="k", linestyle="--")
ax.set_xlim((0, 0.2))
ax.set_ylim((0, 100))
ax.set_xlabel(r"distance to fiber (mm)")
ax.set_ylabel(r"stiffness reduction (%)")
ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
ax.set_yticks([0, 25, 50, 75, 100])
# plt.savefig("img/prior-samples_all.png", bbox_inches="tight")
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

for ip, ipoint in enumerate(ipoints):
    dist = distances[ip]
    idx_l = int(dist / np.max(x) * (len(x) - 1))
    idx_r = idx_l + 1

    x_l = x[idx_l]
    x_r = x[idx_r]
    d_l = true_damage[idx_l] / 100
    d_r = true_damage[idx_r] / 100

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
    cmin=0.0,
    cmax=9120.0,
    colormap="viridis",
    # fname="img/true-field",
)


for i, damage_sample in enumerate(damage_samples):

    fig, ax = plt.subplots()
    for j in range(i):
        ax.plot(x, damage_samples[j], color=color, alpha=opacity)
    ax.plot(x, damage_sample, color=color)
    # ax.plot(x, true_damage, color="k", linestyle="--")
    ax.set_xlim((0, 0.2))
    ax.set_ylim((0, 100))
    ax.set_xlabel(r"distance to fiber (mm)")
    ax.set_ylabel(r"stiffness reduction (%)")
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_yticks([0, 25, 50, 75, 100])
    # plt.savefig("img/prior-samples_sample-{}.png".format(i), bbox_inches="tight")
    plt.show()

    for ip, ipoint in enumerate(ipoints):
        dist = distances[ip]
        idx_l = int(dist / np.max(x) * (len(x) - 1))
        idx_r = idx_l + 1

        x_l = x[idx_l]
        x_r = x[idx_r]
        d_l = damage_sample[idx_l] / 100
        d_r = damage_sample[idx_r] / 100

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
        cmin=0.0,
        cmax=9120.0,
        colormap="viridis",
        # fname="img/prior-field_sample-{}".format(i),
    )

    # eps_xx, eps_yy, gamma_xy = misc.calc_strains(globdat)

    # ElemViewer(
    #     eps_xx[:, 0],
    #     globdat,
    #     maxcolor=0.01,
    #     title=r"max strain, $N_e = {}$".format(len(elems)),
    # )

import numpy as np
import matplotlib.pyplot as plt

from myjive.fem import Tri3Shape
from fem.jive import CJiveRunner
from fem.meshing import read_mesh
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

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)
print("Reading distances from cache")
distances = caching.read_cache(path)

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
plt.show()

##########################
# get integration points #
##########################

props = get_fem_props()
fname = props["userinput"]["gmsh"]["file"]
nodes, elems, groups = read_mesh(fname, read_groups=True)
egroup = groups["matrix"]
shape = Tri3Shape("Gauss3")

name = "ipoints"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)
print("Reading ipoints from cache")
ipoints = caching.read_cache(path)

#######################
# get fiber distances #
#######################

name = "distances"
dependencies = {"nfib": n_fiber, "h": h}
path = caching.get_cache_fpath(name, dependencies)
print("Reading distances from cache")
distances = caching.read_cache(path)

###################
# get stiffnesses #
###################

backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = np.zeros(ipoints.shape[0])

from myjivex.util import ElemViewer


def calc_strains(globdat):
    state0 = globdat["state0"]
    elems = globdat["elemSet"]
    dofs = globdat["dofSpace"]
    shape = globdat["shape"]

    nodes = elems.get_nodes()

    ipcount = shape.ipoint_count()
    nodecount = shape.node_count()
    strains_xx = np.zeros((len(elems), ipcount))
    strains_yy = np.zeros((len(elems), ipcount))
    strains_xy = np.zeros((len(elems), ipcount))

    for ielem, inodes in enumerate(elems):
        idofs = dofs.get_dofs(inodes, ["dx", "dy"])
        eldisp = state0[idofs]

        coords = nodes[inodes]
        grads, wts = shape.get_shape_gradients(coords)

        for ip in range(ipcount):
            B = np.zeros((3, 2 * nodecount))

            for n in range(nodecount):
                B[0, 2 * n + 0] = grads[ip, 0, n]
                B[1, 2 * n + 1] = grads[ip, 1, n]
                B[2, 2 * n + 0] = grads[ip, 1, n]
                B[2, 2 * n + 1] = grads[ip, 0, n]

            ipstrain = B @ eldisp

            strains_xx[ielem, ip] = ipstrain[0]
            strains_yy[ielem, ip] = ipstrain[1]
            strains_xy[ielem, ip] = ipstrain[2]

    return strains_xx, strains_yy, strains_xy


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

    for group_name, egroup in groups.items():
        if group_name == "matrix":
            for ie, ielem in enumerate(egroup):
                ip_stiffness = backdoor["e"][3 * ie : 3 * (ie + 1)]
                elem_stiffness[ielem] = np.mean(ip_stiffness)
        elif group_name == "fiber":
            ielems = egroup.get_indices()
            elem_stiffness[ielems] = 0
        else:
            assert False

    jive = CJiveRunner(props, elems=elems)
    globdat = jive(**backdoor)

    eps_xx, eps_yy, gamma_xy = calc_strains(globdat)

    ElemViewer(
        elem_stiffness,
        globdat,
        maxcolor=E_matrix,
        title=r"stiffness, $N_e = {}$".format(len(elems)),
    )
    # ElemViewer(
    #     eps_xx[:, 0],
    #     globdat,
    #     maxcolor=0.01,
    #     title=r"max strain, $N_e = {}$".format(len(elems)),
    # )

import numpy as np
from scipy.sparse import eye
import matplotlib.pyplot as plt

from myjive.fem import Tri3Shape
from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from probability import Likelihood
from probability.multivariate import Gaussian, SymbolicCovariance
from probability.process import GaussianProcess, ZeroMeanFunction, SquaredExponential
from util.linalg import Matrix

from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params, sampler

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

domain = np.linspace(0.0, 0.2, 101)

inf_prior = GaussianProcess(
    mean=ZeroMeanFunction(),
    cov=SquaredExponential(l=0.02, sigma=2.0),
)

std_pd = 1e-6
cov = inf_prior.calc_cov(domain, domain) + std_pd**2 * np.identity(len(domain))
prior = Gaussian(mean=None, cov=cov, use_scipy_latent=False)

rng = np.random.default_rng(0)

E_matrix = params.material_params["E_matrix"]
alpha = params.material_params["alpha"]
beta = params.material_params["beta"]
c = params.material_params["c"]
d = params.material_params["d"]

saturation = misc.saturation(domain, alpha, beta, c)
damage = misc.damage(saturation, d)
E = E_matrix * (1 - damage)

fig, ax = plt.subplots()

damage_samples = []

for i in range(20):
    sample = prior.calc_sample(rng)
    damage_sample = misc.sigmoid(sample, 1.0, 0.0)
    damage_samples.append(damage_sample)
    ax.plot(domain, damage_sample, color="C0", alpha=0.5)

damage_samples = np.array(damage_samples)

ax.plot(domain, damage, color="k")
ax.set_title("Prior samples of damage field")
plt.show()

##########################
# get integration points #
##########################

props = get_fem_props()
fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)
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

#########################
# get speckle locations #
#########################

obs_size = params.geometry_params["obs_size"]
n_speckle = params.geometry_params["n_speckle"]
r_speckle = params.geometry_params["r_speckle"]
tol = params.geometry_params["tol_speckle"]
seed = seed = params.geometry_params["seed_speckle"]

name = "speckles"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)
print("Reading speckles from cache")
speckles = caching.read_cache(path)

#########################
# get speckle neighbors #
#########################

name = "connectivity"
dependencies = {"nobs": n_speckle}
path = caching.get_cache_fpath(name, dependencies)
print("Reading connectivity from cache")
connectivity = caching.read_cache(path)

############################
# get observation operator #
############################

name = "observer"
dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber, "sparse": True}
path = caching.get_cache_fpath(name, dependencies)
print("Reading observer from cache")
obs_operator = caching.read_cache(path)

####################
# get ground truth #
####################

name = "truth"
dependencies = {"nobs": n_speckle, "nfib": n_fiber}
path = caching.get_cache_fpath(name, dependencies)
print("Reading truth from cache")
truth = caching.read_cache(path)


class CustomLikelihood(Likelihood):

    def __init__(self):
        self.ipoints = ipoints
        self.distances = distances
        self.operator = obs_operator
        self.observations = truth
        n_obs = len(self.observations)
        std = 1e-6
        self.noise = SymbolicCovariance(Matrix(std**2 * eye(n_obs)))
        self.dist = Gaussian(self.observations, self.noise)

        self._input_map = (len(domain) - 1) / np.max(domain)

    def calc_logpdf(self, x):
        damage = misc.sigmoid(x, 1.0, 0.0)

        for ip, ipoint in enumerate(ipoints):
            dist = distances[ip]
            idx_l = int(dist * self._input_map)
            idx_r = idx_l + 1

            x_l = domain[idx_l]
            x_r = domain[idx_r]
            d_l = damage[idx_l]
            d_r = damage[idx_r]

            assert x_l <= dist <= x_r

            dam = d_l + (dist - x_l) / (x_r - x_l) * (d_r - d_l)
            backdoor["e"][ip] = E_matrix * (1 - dam)

        jive = CJiveRunner(props, elems=elems, egroups=groups)
        globdat = jive(**backdoor)

        state0 = globdat["state0"]
        pred = self.operator @ state0

        loglikelihood = self.dist.calc_logpdf(pred)
        return loglikelihood


likelihood = CustomLikelihood()

n_burn = 1000
n_sample = 5000


def linear_tempering(i):
    if i < n_burn:
        return i / n_burn
    else:
        return 1.0


sampler = sampler.EllipticalSliceSampler(
    prior=prior,
    likelihood=likelihood,
    n_sample=n_sample,
    n_burn=n_burn,
    seed=0,
    tempering=linear_tempering,
    return_info=True,
)

samples, info = sampler()

fname = "posterior-samples_h-{:.3f}.npy".format(h)
np.save(fname, samples)

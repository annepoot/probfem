import os
import numpy as np
from scipy.sparse import csr_array
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet

from experiments.reproduction.nonhierarchical.pullout_bar.props import get_fem_props
from fem.jive import CJiveRunner
from fem.meshing import (
    mesh_interval_with_line2,
    create_phi_from_globdat,
    create_hypermesh,
)
from probability.multivariate import Gaussian
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from util.linalg import Matrix
from bfem.observation import compute_bfem_observations

props = get_fem_props()


def u_exact(x):
    k = props["model"]["model"]["spring"]["k"]
    E = props["model"]["model"]["elastic"]["material"]["E"]
    f = props["model"]["model"]["neum"]["initLoad"]

    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))

    return A * (np.exp(nu * x) + np.exp(-nu * x))


def invert_mesh(mesh):
    if isinstance(mesh, ElementSet):
        elems = mesh
        nodes = elems.get_nodes()
    else:
        nodes, elems = mesh
    assert isinstance(nodes, NodeSet)
    assert isinstance(elems, ElementSet)

    coords = nodes.get_coords()

    left_boundary = np.min(coords, axis=0)
    right_boundary = np.max(coords, axis=0)

    inv_nodes = XNodeSet()
    inv_nodes.add_node(left_boundary)
    for inodes in elems:
        midpoint = np.mean(nodes[inodes], axis=0)
        inv_nodes.add_node(midpoint)
    inv_nodes.add_node(right_boundary)
    inv_nodes.to_nodeset()

    inv_coords = inv_nodes.get_coords()
    sort_idx = np.argsort(inv_coords[:, 0], axis=0)

    inv_elems = XElementSet(inv_nodes)
    for ielem in np.arange(len(inv_nodes) - 1):
        inodes = np.array([sort_idx[ielem], sort_idx[ielem + 1]])
        inv_elems.add_element(inodes)
    inv_elems.to_elementset()

    return inv_nodes, inv_elems


def dropout_mesh(*, n, seed):
    n_ref = 2048
    ref_coords = np.linspace(0, 1, n_ref + 1)

    rng = np.random.default_rng(seed)
    idx_set = np.arange(1, n_ref)
    rng.shuffle(idx_set)

    coords = np.zeros((n + 1, 1))
    coords[0, 0] = 0.0
    coords[n, 0] = 1.0
    coords[1:n, 0] = np.sort(ref_coords[idx_set[: n - 1]])

    nodes = XNodeSet()
    for coord in coords:
        nodes.add_node(coord)
    nodes.to_nodeset()

    elems = XElementSet(nodes)
    for ielem in np.arange(len(nodes) - 1):
        inodes = np.array([ielem, ielem + 1])
        elems.add_element(inodes)
    elems.to_elementset()

    return nodes, elems


def random_mesh(*, n, seed):
    rng = np.random.default_rng(seed)
    coords = np.zeros((n + 1, 1))
    coords[0, 0] = 0.0
    coords[n, 0] = 1.0
    coords[1:n, 0] = np.sort(rng.uniform(size=n - 1))

    nodes = XNodeSet()
    for coord in coords:
        nodes.add_node(coord)
    nodes.to_nodeset()

    elems = XElementSet(nodes)
    for ielem in np.arange(len(nodes) - 1):
        inodes = np.array([ielem, ielem + 1])
        elems.add_element(inodes)
    elems.to_elementset()

    return nodes, elems


def calc_norm(obs_elems, ref_elems):
    (hyp_nodes, hyp_elems), hyp_map = create_hypermesh(obs_elems, ref_elems)

    module_props = get_fem_props()

    jive = CJiveRunner(module_props, elems=obs_elems)
    globdat = jive()
    u_obs = globdat["state0"]
    K_obs = globdat["matrix0"]
    n_obs = len(u_obs)
    alpha2_mle = u_obs @ K_obs @ u_obs / n_obs

    plot_nodes, plot_elems = mesh_interval_with_line2(n=2048)

    module_props = get_fem_props()
    plot_jive_runner = CJiveRunner(module_props, elems=plot_elems)
    plot_globdat = plot_jive_runner()

    model_props = module_props.pop("model")
    ref_jive_runner = CJiveRunner(module_props, elems=ref_elems)
    obs_jive_runner = CJiveRunner(module_props, elems=obs_elems)
    hyp_jive_runner = CJiveRunner(module_props, elems=hyp_elems)

    inf_cov = InverseCovarianceOperator(model_props=model_props, scale=alpha2_mle)
    inf_prior = GaussianProcess(None, inf_cov)
    obs_prior = ProjectedPrior(prior=inf_prior, jive_runner=obs_jive_runner)
    obs_globdat = obs_prior.globdat
    ref_prior = ProjectedPrior(prior=inf_prior, jive_runner=ref_jive_runner)
    ref_globdat = ref_prior.globdat
    hyp_prior = ProjectedPrior(prior=inf_prior, jive_runner=hyp_jive_runner)
    hyp_globdat = hyp_prior.globdat

    H_obs, f_obs = compute_bfem_observations(obs_prior, hyp_prior)
    H_ref, f_ref = compute_bfem_observations(ref_prior, hyp_prior)

    Phi_obs = H_obs[0].T
    Phi_ref = H_ref[0].T
    K_hyp = H_obs[1]

    K_obs = Matrix((Phi_obs.T @ K_hyp @ Phi_obs).evaluate(), name="K_obs")
    K_ref = Matrix((Phi_ref.T @ K_hyp @ Phi_ref).evaluate(), name="K_ref")
    K_x = Matrix((Phi_ref.T @ K_hyp @ Phi_obs).evaluate(), name="K_x")

    Phi_plot = create_phi_from_globdat(hyp_globdat, plot_globdat)
    Phi_plot = Matrix(Phi_plot, name="Phi_p")

    n = len(plot_nodes) - 1
    h = 1 / n
    base_idx = np.arange(n)

    M_plot_rowidx = np.concatenate((base_idx, base_idx + 1, base_idx, base_idx + 1))
    M_plot_colidx = np.concatenate((base_idx, base_idx + 1, base_idx + 1, base_idx))
    M_plot_values = np.concatenate((np.full(2 * n, h / 3), np.full(2 * n, h / 6)))
    M_plot = csr_array((M_plot_values, (M_plot_rowidx, M_plot_colidx)))

    M_plot = Matrix(M_plot, name="M_plot")
    M_hyp = Matrix((Phi_plot.T @ M_plot @ Phi_plot).evaluate(), name="M_hyp")
    M_ref = Matrix((Phi_ref.T @ M_hyp @ Phi_ref).evaluate(), name="M_ref")
    M_obs = Matrix((Phi_obs.T @ M_hyp @ Phi_obs).evaluate(), name="M_obs")
    M_x = Matrix((Phi_ref.T @ M_hyp @ Phi_obs).evaluate(), name="M_x")

    prior_norm_ref = np.trace((K_ref.inv @ M_ref).evaluate())
    prior_norm_obs = np.trace((K_obs.inv @ M_obs).evaluate())

    posterior_norm = prior_norm_ref + prior_norm_obs
    posterior_norm -= 2 * np.trace((K_ref.inv @ K_x @ K_obs.inv @ M_x.T).evaluate())

    return posterior_norm


n_elem = 4

# options: exact, hierarchical, inverted, random
ref_type = "inverted"

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

if ref_type == "exact":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=2048)
elif ref_type == "hierarchical":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=4 * n_elem)
elif ref_type == "inverted":
    ref_nodes, ref_elems = invert_mesh(obs_elems)
elif ref_type == "random":
    ref_nodes, ref_elems = random_mesh(n=n_elem, seed=0)
else:
    assert False

# options: exact, hierarchical, inverted, random
n_obs = 4
n_refs = np.array([4, 8, 16, 32, 64, 128, 192, 256])
n_seed = 10

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

hierarchical_norms = np.zeros_like(n_refs, dtype=float)
inverted_norms = np.zeros_like(n_refs, dtype=float)
random_norms = np.zeros((len(n_refs), n_seed), dtype=float)
dropout_norms = np.zeros((len(n_refs), n_seed), dtype=float)

for ref_type in ["exact", "hierarchical", "inverted", "dropout", "random"]:
    if ref_type == "exact":
        ref_nodes, ref_elems = mesh_interval_with_line2(n=2048)
        exact_norm = calc_norm(obs_elems, ref_elems)
    elif ref_type == "hierarchical":
        for i, n_ref in enumerate(n_refs):
            ref_nodes, ref_elems = mesh_interval_with_line2(n=n_ref)
            hierarchical_norms[i] = calc_norm(obs_elems, ref_elems)
    elif ref_type == "inverted":
        ref_nodes, ref_elems = invert_mesh(obs_elems)
        inverted_norm = calc_norm(obs_elems, ref_elems)
    elif ref_type == "dropout":
        for i, n_ref in enumerate(n_refs):
            for j in range(n_seed):
                ref_nodes, ref_elems = dropout_mesh(n=n_ref, seed=j)
                dropout_norms[i, j] = calc_norm(obs_elems, ref_elems)
    elif ref_type == "random":
        for i, n_ref in enumerate(n_refs):
            for j in range(n_seed):
                ref_nodes, ref_elems = random_mesh(n=n_ref, seed=j)
                random_norms[i, j] = calc_norm(obs_elems, ref_elems)
    else:
        assert False


fig, ax = plt.subplots()
ax.scatter(
    n_refs,
    exact_norm - hierarchical_norms,
    color="C0",
    marker=".",
    label="hierarchical",
)
ax.scatter(
    n_obs + 1, exact_norm - inverted_norm, color="C1", marker=".", label="inverted"
)

for j in range(n_seed):
    label = "random" if j == 0 else None
    ax.scatter(
        n_refs, exact_norm - random_norms[:, j], color="C2", marker=".", label=label
    )

# for j in range(n_seed):
#     label = "dropout" if j == 0 else None
#     ax.scatter(
#         n_refs, exact_norm - dropout_norms[:, j], color="C3", marker=".", label=label
#     )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_aspect("equal")
ax.legend()
plt.show()

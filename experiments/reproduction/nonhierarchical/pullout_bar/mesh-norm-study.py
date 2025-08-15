import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

from fem.meshing import mesh_interval_with_line2

from experiments.reproduction.nonhierarchical.pullout_bar import misc


n_elem = 4

# options: exact, hierarchical, dual, random
ref_type = "dual"

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

if ref_type == "exact":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=2048)
elif ref_type == "hierarchical":
    ref_nodes, ref_elems = mesh_interval_with_line2(n=4 * n_elem)
elif ref_type == "dual":
    ref_nodes, ref_elems = misc.dual_mesh(obs_elems)
elif ref_type == "random":
    ref_nodes, ref_elems = misc.random_mesh(n=n_elem, seed=0)
else:
    assert False

# options: exact, hierarchical, dual, random
n_obs = 4
n_refs = np.array([4, 8, 16, 32, 64, 128, 256])
n_refs_hierarchical = n_refs[1:]
n_refs_random = np.concatenate([n + n // 4 * np.array([0, 1, 2, 3]) for n in n_refs])
n_refs_random = np.concatenate([np.array([1, 2, 3]), n_refs_random])
n_refs_optimal = np.array([1, 2, 3, 4, 5, 7, 9, 13, 17, 33, 65, 129, 257])
n_seed = 10

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

hierarchical_norms = np.zeros_like(n_refs_hierarchical, dtype=float)
dual_norms = np.zeros_like(n_refs, dtype=float)
random_norms = np.zeros((len(n_refs_random), n_seed), dtype=float)
dropout_norms = np.zeros((len(n_refs_random), n_seed), dtype=float)
optimal_norms = np.zeros_like(n_refs_optimal, dtype=float)

for ref_type in ["exact", "hierarchical", "dual", "optimal", "random"]:
    if ref_type == "exact":
        ref_nodes, ref_elems = mesh_interval_with_line2(n=2048)
        exact_norm = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "hierarchical":
        for i, n_ref in enumerate(n_refs_hierarchical):
            ref_nodes, ref_elems = mesh_interval_with_line2(n=n_ref)
            hierarchical_norms[i] = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "dual":
        for i, n_ref in enumerate(n_refs):
            ref_nodes, ref_elems = mesh_interval_with_line2(n=n_ref)
            ref_nodes, ref_elems = misc.dual_mesh(ref_elems)
            dual_norms[i] = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "dropout":
        for i, n_ref in enumerate(n_refs):
            for j in range(n_seed):
                ref_nodes, ref_elems = misc.dropout_mesh(n=n_ref, seed=j)
                dropout_norms[i, j] = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "random":
        for i, n_ref in enumerate(n_refs_random):
            for j in range(n_seed):
                ref_nodes, ref_elems = misc.random_mesh(n=n_ref, seed=j)
                random_norms[i, j] = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "optimal":
        for i, n_ref in enumerate(n_refs_optimal):
            if n_ref == 1:
                continue

            ref_nodes, ref_elems = misc.optimal_mesh(obs_elems=obs_elems, n=n_ref)
            optimal_norms[i] = misc.calc_norm(obs_elems, ref_elems)
    else:
        assert False


fig, ax = plt.subplots()

for j in range(n_seed):
    label = "random" if j == 0 else None
    ax.plot(
        n_refs_random,
        exact_norm - random_norms[:, j],
        color="0.5",
        marker=".",
        label=label,
        zorder=1,
    )

ax.scatter(
    n_refs_hierarchical,
    exact_norm - hierarchical_norms,
    color="C0",
    marker="o",
    label="hierarchical",
)
ax.scatter(
    n_refs + 1,
    exact_norm - dual_norms,
    color="C1",
    marker="o",
    label="dual",
)
ax.plot(
    n_refs_optimal,
    exact_norm - optimal_norms,
    color="k",
    label="bounds",
)
ax.axhline(
    exact_norm,
    color="k",
    label="",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$d - \tilde{d}$")
ax.set_xticks(n_refs)
ax.xaxis.set_major_locator(LogLocator(base=2))
ax.xaxis.set_major_formatter(LogFormatter(base=2))
ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(1.25, 1.5, 1.75)))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_aspect("equal")
ax.legend()
fname = os.path.join("img", "convergence-study.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
n_refs_hierarchical = n_refs
n_refs_random = np.concatenate([n + n // 4 * np.array([0, 1, 2, 3]) for n in n_refs])
n_refs_random = np.concatenate([np.array([1, 2, 3]), n_refs_random])
n_refs_optimal = np.array([1, 2, 3, 4, 5, 7, 9, 13, 17, 33, 65, 129, 257])
n_seed = 10

obs_nodes, obs_elems = mesh_interval_with_line2(n=n_elem)

hierarchical_norms = np.zeros_like(n_refs_hierarchical, dtype=float)
dual_norms = np.zeros_like(n_refs, dtype=float)
left_norms = np.zeros_like(n_refs, dtype=float)
right_norms = np.zeros_like(n_refs, dtype=float)
random_norms = np.zeros((len(n_refs_random), n_seed), dtype=float)
dropout_norms = np.zeros((len(n_refs_random), n_seed), dtype=float)
optimal_norms = np.zeros_like(n_refs_optimal, dtype=float)

for ref_type in ["exact", "hierarchical", "dual", "left", "right", "optimal", "random"]:
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
    elif ref_type == "left":
        for i, n_ref in enumerate(n_refs):
            ref_nodes, ref_elems = mesh_interval_with_line2(n=n_ref + 1)
            ref_nodes._data[: n_ref + 1, 0] = np.linspace(0, 0.5, n_ref + 1)
            left_norms[i] = misc.calc_norm(obs_elems, ref_elems)
    elif ref_type == "right":
        for i, n_ref in enumerate(n_refs):
            ref_nodes, ref_elems = mesh_interval_with_line2(n=n_ref + 1)
            ref_nodes._data[1 : n_ref + 2, 0] = np.linspace(0.5, 1.0, n_ref + 1)
            right_norms[i] = misc.calc_norm(obs_elems, ref_elems)
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


highlight = sns.color_palette("rocket_r", n_colors=8)[int(np.log2(n_obs))]
major_ticks = 2 ** np.arange(0, 10)
minor_ticks = np.array([3])
for offset in [1.25, 1.50, 1.75]:
    minor_ticks = np.concatenate((minor_ticks, major_ticks[2:-1] * offset))


fig, ax = plt.subplots()

for n_ref, norm in zip(n_refs_hierarchical, hierarchical_norms):
    c = highlight if n_ref == 8 else "0.3"
    label = "hierarchical" if n_ref == 16 else None
    ax.scatter(n_ref, norm, color=c, marker="o", facecolor="none", label=label)

for n_ref, norm in zip(n_refs + 1, dual_norms):
    c = highlight if n_ref == 5 else "0.3"
    label = "dual" if n_ref == 9 else None
    ax.scatter(n_ref, norm, color=c, marker="x", label=label)

for n_ref, norm in zip(n_refs + 1, left_norms):
    c = highlight if n_ref == 5 else "0.3"
    label = "left/right" if n_ref == 9 else None
    ax.scatter(n_ref, norm, color=c, marker="+", label=label)

for j in range(n_seed):
    label = "random" if j == 0 else None
    ax.plot(
        n_refs_random,
        random_norms[:, j],
        color="0.8",
        marker=".",
        label=label,
        zorder=0,
    )

ax.plot(
    n_refs_optimal,
    optimal_norms,
    color="k",
    linestyle="--",
    label="optimized",
    zorder=0,
)
ax.axhline(exact_norm, color="k", label="bounds")
ax.axhline(0.0, color="k")

ax.set_xscale("log")
ax.set_yscale("linear")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\tilde{d}$")
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels([str(m) for m in major_ticks])
ax.set_xticklabels([], minor=True)
# ax.legend()
fname = os.path.join("img", "convergence-absolute.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()


fig, ax = plt.subplots()

for n_ref, norm in zip(n_refs_hierarchical, hierarchical_norms):
    c = highlight if n_ref == 8 else "0.3"
    label = "hierarchical" if n_ref == 16 else None
    ax.scatter(
        n_ref, exact_norm - norm, color=c, marker="o", facecolor="none", label=label
    )

for n_ref, norm in zip(n_refs + 1, dual_norms):
    c = highlight if n_ref == 5 else "0.3"
    label = "dual" if n_ref == 9 else None
    ax.scatter(n_ref, exact_norm - norm, color=c, marker="x", label=label)

for n_ref, norm in zip(n_refs + 1, left_norms):
    c = highlight if n_ref == 5 else "0.3"
    label = "left/right" if n_ref == 9 else None
    ax.scatter(n_ref, exact_norm - norm, color=c, marker="+", label=label)

for j in range(n_seed):
    label = "random" if j == 0 else None
    ax.plot(
        n_refs_random,
        exact_norm - random_norms[:, j],
        color="0.8",
        marker=".",
        label=label,
        zorder=0,
    )

ax.plot(
    n_refs_optimal,
    exact_norm - optimal_norms,
    color="k",
    linestyle="--",
    label="optimized",
    zorder=0,
)
ax.axhline(exact_norm, color="k", label="bounds")

x, y = 48, 0.01
trix, triy = [x, x * 2, x, x], [y, y / 2, y / 2, y]
ax.plot(trix, triy, color="k")
ax.text(x * 0.9, y / np.sqrt(2), "$1$", ha="right", va="center_baseline")
ax.text(x * np.sqrt(2), y / 2 * 0.9, "$1$", ha="center", va="top")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$d - \tilde{d}$")
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels([str(m) for m in major_ticks])
ax.set_xticklabels([], minor=True)
ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
fname = os.path.join("img", "convergence-relative.pdf")
plt.savefig(fname=fname, bbox_inches="tight")
plt.show()

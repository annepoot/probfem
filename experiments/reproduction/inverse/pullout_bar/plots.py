import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fem.jive import CJiveRunner
from fem.meshing import mesh_interval_with_line2, create_phi_from_globdat
from probability.multivariate import ConditionedGaussian

from experiments.reproduction.inverse.pullout_bar.props import get_fem_props

# matplotlib settings
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"


def u_exact(x):
    props = get_fem_props()
    k = props["model"]["model"]["spring"]["k"]
    E = props["model"]["model"]["elastic"]["material"]["E"]
    f = props["model"]["model"]["neum"]["initLoad"]

    nu = np.sqrt(k / E)
    eps = f / E

    A = eps / (nu * (np.exp(nu) - np.exp(-nu)))
    return A * (np.exp(nu * x) + np.exp(-nu * x))


def exact_plot(n_elems, us):
    u_obs = np.array([u[-1] for u in us])
    e_obs = u_exact(1) - u_obs
    
    xmarkers = np.linspace(0.0, 1.0, 6)
    ymarkers = np.linspace(-0.25, 1.5, 8)
    colors = sns.color_palette("rocket_r", n_colors=8)
    
    plt.figure()
    for i, (n_elem, u) in enumerate(zip(n_elems, us)):
        x = np.linspace(0, 1, n_elem + 1)
        plt.plot(x, u, label=r"$\sfrac{1}{" + str(n_elem) + "}$", color=colors[i])
    plt.plot(x, u_exact(x), color="k", label="truth")
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u$", fontsize=12)
    plt.xticks(xmarkers)
    plt.yticks(ymarkers)
    plt.ylim(ymarkers[[0, -1]])
    legend = plt.legend(title=r"$h$")
    fontsize = "12"
    plt.setp(legend.get_texts(), fontsize=fontsize)
    plt.setp(legend.get_title(), fontsize=fontsize)
    plt.show()


def bfem_plot(ref_dist):
    props = get_fem_props()
    plot_nodes, plot_elems = mesh_interval_with_line2(n=2880)
    plot_jive_runner = CJiveRunner(props, elems=plot_elems)
    plot_globdat = plot_jive_runner()

    if isinstance(ref_dist, ConditionedGaussian):
        dist_type = "posterior"
    else:
        dist_type = "prior"

    if dist_type == "posterior":
        globdat = ref_dist.prior.globdat
    else:
        globdat = ref_dist.globdat

    Phi_plot = create_phi_from_globdat(globdat, plot_globdat)
    plot_dist =  ref_dist @ Phi_plot.T
    
    samples = plot_dist.calc_samples(20, 0)
    mean = plot_dist.calc_mean()
    std = plot_dist.calc_std()

    c = sns.color_palette("rocket_r", n_colors=8)[2]
    x_plot = np.linspace(0, 1, len(mean))
    
    xmarkers = np.linspace(0.0, 1.0, 6)

    if dist_type == "posterior":
        ymarkers = np.linspace(-1.0, 2.0, 7)
    else:
        ymarkers = np.linspace(-1.5, 1.5, 7)
    
    plt.figure()
    plt.plot(x_plot, mean, color=c)
    plt.plot(x_plot, samples.T, color=c, linewidth=0.5)
    plt.fill_between(x_plot, mean - 2 * std, mean + 2 * std, color=c, alpha=0.3)
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u$", fontsize=12)
    plt.xticks(xmarkers)
    plt.yticks(ymarkers)
    plt.ylim(ymarkers[[0, -1]])
    plt.show()


def rmfem_plot(xs, us):
    props = get_fem_props()
    pert_nodes, pert_elems = mesh_interval_with_line2(n=len(xs[0]) - 1)
    jive = CJiveRunner(props, elems=pert_elems)

    plot_nodes, plot_elems = mesh_interval_with_line2(n=2880)
    plot_jive_runner = CJiveRunner(props, elems=plot_elems)
    plot_globdat = plot_jive_runner()

    x_plot = plot_nodes.get_coords().flatten()
    u_plots = []

    for x, u in zip(xs, us):
        pert_nodes._data[:, :] = x
        jive.update_elems(pert_elems)
        globdat_pert = jive()

        Phi_plot = create_phi_from_globdat(globdat_pert, plot_globdat)
        u_plot = Phi_plot @ u
        u_plots.append(u_plot)

    u_plots = np.array(u_plots)

    mean = np.mean(u_plots, axis=0)
    std = np.std(u_plots, axis=0)
    
    c = sns.color_palette("rocket_r", n_colors=8)[2]
    xmarkers = np.linspace(0.0, 1.0, 6)
    ymarkers = np.linspace(-1.0, 2.0, 7)
    
    plt.figure()
    plt.plot(x_plot, mean, color=c)
    plt.plot(x_plot, u_plots[:20].T, color=c, linewidth=0.5)
    plt.fill_between(x_plot, mean - 2 * std, mean + 2 * std, color=c, alpha=0.3)
    plt.plot(x_plot, u_exact(x_plot), color="k")
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$u$", fontsize=12)
    plt.xticks(xmarkers)
    plt.yticks(ymarkers)
    plt.ylim(ymarkers[[0, -1]])
    plt.show()


def scatter_plot(df):
    variables = ["E", "k"]
    refs_by_var = {
        "E": 0.8,
        "k": 70.0,
    }

    labels_by_var = {
        "E": r"$EA$",
        "k": r"$k$",
    }

    def lims_by_var(width):
        lims_by_var = {}
        for var in variables:
            ref = refs_by_var[var]
            lims_by_var[var] = (ref * (1 - width), ref * (1 + width))
        return lims_by_var

    width = 1.0
    n_elem_range = np.array(df["n_elem"].drop_duplicates())
    colors = dict(zip([1, 2, 4, 8, 16, 32, 64], sns.color_palette("rocket_r", n_colors=8)))

    df["n_elem"] = df["n_elem"].astype(str)
    df["h"] = r"\sfrac{1}{" + df["n_elem"] + "}"
    
    rng = np.random.default_rng(0)
    prior_mean = np.log(np.array([1.0, 100.0]))
    prior_cov = 0.1**2 * np.identity(2)
    n_sample = len(df) // len(n_elem_range)
    samples = rng.multivariate_normal(prior_mean, prior_cov, size=n_sample)
    E_prior = np.exp(samples[:, 0])
    k_prior = np.exp(samples[:, 1])

    fig, ax = plt.subplots(figsize=(4, 4))

    plot = sns.scatterplot(
        data=df,
        x="E",
        y="k",
        hue="h",
        alpha=0.6,
        marker=".",
        linewidths=0.0,
        ax=ax,
        palette=[colors[n_elem] for n_elem in n_elem_range],
    )

    ref_E = refs_by_var["E"]
    ref_k = refs_by_var["k"]
    lims_E = lims_by_var(width)["E"]
    lims_k = lims_by_var(width)["k"]

    ax.set_xlim(lims_E)
    ax.set_ylim(lims_k)
    ax.set_xticks(np.linspace(lims_E[0], lims_E[1], 5))
    ax.set_yticks(np.linspace(lims_k[0], lims_k[1], 5))
    ax.xaxis.set_label_text(labels_by_var["E"])
    ax.yaxis.set_label_text(labels_by_var["k"])

    ax.scatter(
        E_prior, k_prior, c="0.7", marker=".", alpha=0.4, zorder=0, lw=0, label="prior"
    )
    E_left = np.linspace(lims_E[0] + 1e-8, ref_E, 500)[:-1]
    E_right = np.linspace(ref_E, lims_E[1], 500)
    E = np.concatenate((E_left, E_right))
    k = ref_E * ref_k / E
    ax.plot(E, k, ":ko", markevery=[500], markersize=5, label="truth")
    legend = ax.legend(title=r"$h$", loc="upper left")
    fontsize = "12"
    plt.setp(legend.get_texts(), fontsize=fontsize)
    plt.setp(legend.get_title(), fontsize=fontsize)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    plt.show()

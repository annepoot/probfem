import numpy as np
from scipy.sparse import csr_array
from scipy.optimize import minimize, Bounds
from warnings import warn

from myjive.fem import NodeSet, XNodeSet, ElementSet, XElementSet

from bfem import compute_bfem_observations
from fem.meshing import (
    create_phi_from_globdat,
    create_hypermesh,
    mesh_interval_with_line2,
)
from fem.jive import CJiveRunner
from probability.process import (
    GaussianProcess,
    InverseCovarianceOperator,
    ProjectedPrior,
)
from util.linalg import Matrix

from experiments.reproduction.nonhierarchical.pullout_bar.props import get_fem_props


def u_exact(x, *, k, E, f):
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


def optimal_mesh(*, obs_elems, n):

    if n == 2:
        hard_coords = np.array([0.12536113])
    elif n == 3:
        hard_coords = np.array([0.12540454, 0.87455606])
    elif n == 4:
        hard_coords = np.array([0.15917673, 0.34521442, 0.87063136])
    elif n == 5:
        hard_coords = np.array([0.14652817, 0.37939153, 0.62049205, 0.85318558])
    elif n == 7:
        hard_coords = np.array(
            [0.14513986, 0.33825499, 0.42115435, 0.57844916, 0.66149548, 0.854663]
        )
    elif n == 9:
        # fmt: off
        hard_coords = np.array([
            0.10088008, 0.17684051, 0.33441463, 0.41897797, 0.58112921,
            0.66473487, 0.82255262, 0.89874585
        ])
        # fmt: on
    elif n == 13:
        # fmt: off
        hard_coords = np.array([
            0.07534868, 0.13671393, 0.19648175, 0.31150081, 0.37730594,
            0.44281214, 0.55750008, 0.62303031, 0.68847226, 0.80278896,
            0.86230827, 0.92383495
        ])
        # fmt: on
    elif n == 17:
        # fmt: off
        hard_coords = np.array([
            0.07618009, 0.13721325, 0.19678218, 0.2861362 , 0.3310556 ,
            0.37549201, 0.42026113, 0.4641772 , 0.53383201, 0.57811894,
            0.62304933, 0.66796825, 0.71288379, 0.80286916, 0.86233352,
            0.92334549
        ])
        # fmt: on
    elif n == 33:
        # fmt: off
        hard_coords = np.array([
            0.02931296, 0.05729606, 0.08544819, 0.11289215, 0.14111077,
            0.16943384, 0.1977596 , 0.22656395, 0.27374119, 0.30322303,
            0.33300836, 0.36267015, 0.39160278, 0.42015415, 0.44920696,
            0.47826142, 0.52204895, 0.55127351, 0.58056352, 0.60986235,
            0.63915983, 0.66893932, 0.69824222, 0.72703363, 0.7734376 ,
            0.80224649, 0.8305657 , 0.85888711, 0.88645933, 0.91405769,
            0.94140685, 0.97012227
        ])
        # fmt: on
    elif n == 65:
        # fmt: off
        hard_coords = np.array([
            0.01415881, 0.02783355, 0.04153011, 0.05589954, 0.0708424 ,
            0.0864254 , 0.10155688, 0.11673552, 0.13232313, 0.14795107,
            0.16308392, 0.17871184, 0.19385047, 0.20849652, 0.22363226,
            0.23862017, 0.26123016, 0.27587928, 0.29026713, 0.30496726,
            0.32031359, 0.33544865, 0.35107473, 0.36670077, 0.38232204,
            0.3979488 , 0.4135743 , 0.42870945, 0.44375722, 0.45849916,
            0.47363141, 0.48877501, 0.5107394 , 0.52539247, 0.54003973,
            0.55517826, 0.57031859, 0.58593791, 0.60156613, 0.61718783,
            0.63280991, 0.64844096, 0.66406353, 0.67886153, 0.69384819,
            0.70849565, 0.72313923, 0.73827653, 0.76123082, 0.77540987,
            0.79003842, 0.80493791, 0.82031279, 0.83544871, 0.8510741 ,
            0.86621324, 0.88183397, 0.89794838, 0.91357406, 0.9287266 ,
            0.94331949, 0.95752053, 0.97118842, 0.98535326
        ])
        # fmt: on
    elif n == 129:
        # fmt: off
        hard_coords = np.array([
            0.00594058, 0.01217494, 0.01915026, 0.02687194, 0.03465994,
            0.04246385, 0.05026731, 0.05807713, 0.06590128, 0.0737098 ,
            0.08153094, 0.08935472, 0.09717272, 0.10498033, 0.11279503,
            0.12060971, 0.12841924, 0.13623174, 0.14404345, 0.15185809,
            0.15966828, 0.16748158, 0.17529263, 0.18310114, 0.19091415,
            0.19873101, 0.20653983, 0.21434974, 0.22216191, 0.22997791,
            0.23765486, 0.24457292, 0.25473052, 0.26176549, 0.26904417,
            0.27685499, 0.28466742, 0.29247956, 0.30029326, 0.30810029,
            0.31591694, 0.32373057, 0.331543  , 0.33935288, 0.347168  ,
            0.35498068, 0.36279317, 0.37060567, 0.37841816, 0.38623065,
            0.39404316, 0.40185565, 0.40966814, 0.41748064, 0.42529314,
            0.43310563, 0.44091814, 0.44873064, 0.45654315, 0.46435565,
            0.47216917, 0.4799765 , 0.48765783, 0.49414943, 0.50487114,
            0.51181589, 0.51904282, 0.52685556, 0.53466809, 0.54248075,
            0.55029172, 0.55810551, 0.565918  , 0.57373052, 0.58154297,
            0.58935097, 0.59716286, 0.60497642, 0.61279329, 0.62060213,
            0.62841825, 0.63623079, 0.64404303, 0.65185666, 0.65965551,
            0.66748199, 0.67529302, 0.68310457, 0.69092409, 0.69872431,
            0.70654418, 0.71435515, 0.72216763, 0.72997755, 0.73765376,
            0.74456425, 0.75471941, 0.76177027, 0.76904136, 0.77684972,
            0.78466228, 0.79247715, 0.80028989, 0.80810715, 0.81591863,
            0.82372944, 0.83154015, 0.83935085, 0.8471637 , 0.85498067,
            0.86279259, 0.87060459, 0.87841869, 0.8862315 , 0.89404429,
            0.90185729, 0.9096699 , 0.91748238, 0.9252947 , 0.93310651,
            0.94091827, 0.94873216, 0.95654299, 0.96435695, 0.97216046,
            0.97997264, 0.98729994, 0.99364796
        ])
        # fmt: on
    elif n == 257:
        # fmt: off
        hard_coords = np.array([
            0.00195313, 0.00585937, 0.00976562, 0.01367187, 0.01757812,
            0.02148438, 0.02539062, 0.02929688, 0.03320312, 0.03710938,
            0.04101562, 0.04492188, 0.04882812, 0.05273438, 0.05664062,
            0.06054688, 0.06445312, 0.06835938, 0.07226562, 0.07617188,
            0.08007812, 0.08398438, 0.08789062, 0.09179688, 0.09570312,
            0.09960938, 0.10351562, 0.10742188, 0.11132812, 0.11523438,
            0.11914062, 0.12304688, 0.12695312, 0.13085938, 0.13476562,
            0.13867188, 0.14257812, 0.14648438, 0.15039062, 0.15429688,
            0.15820312, 0.16210938, 0.16601562, 0.16992188, 0.17382812,
            0.17773438, 0.18164062, 0.18554688, 0.18945312, 0.19335938,
            0.19726562, 0.20117188, 0.20507812, 0.20898438, 0.21289062,
            0.21679688, 0.22070312, 0.22460938, 0.22851562, 0.23242188,
            0.23632812, 0.24023438, 0.24414062, 0.24804688, 0.25195312,
            0.25585938, 0.25976562, 0.26367188, 0.26757812, 0.27148438,
            0.27539062, 0.27929688, 0.28320312, 0.28710938, 0.29101562,
            0.29492188, 0.29882812, 0.30273438, 0.30664062, 0.31054688,
            0.31445312, 0.31835938, 0.32226562, 0.32617188, 0.33007812,
            0.33398438, 0.33789062, 0.34179688, 0.34570312, 0.34960938,
            0.35351562, 0.35742188, 0.36132812, 0.36523438, 0.36914062,
            0.37304688, 0.37695312, 0.38085938, 0.38476562, 0.38867188,
            0.39257812, 0.39648438, 0.40039062, 0.40429688, 0.40820312,
            0.41210938, 0.41601562, 0.41992188, 0.42382812, 0.42773438,
            0.43164062, 0.43554688, 0.43945312, 0.44335938, 0.44726562,
            0.45117188, 0.45507812, 0.45898438, 0.46289062, 0.46679688,
            0.47070312, 0.47460938, 0.47851562, 0.48242188, 0.48632812,
            0.49023438, 0.49414062, 0.49804688, 0.50195312, 0.50585938,
            0.50976562, 0.51367188, 0.51757812, 0.52148438, 0.52539062,
            0.52929688, 0.53320312, 0.53710938, 0.54101562, 0.54492188,
            0.54882812, 0.55273438, 0.55664062, 0.56054688, 0.56445312,
            0.56835938, 0.57226562, 0.57617188, 0.58007812, 0.58398438,
            0.58789062, 0.59179688, 0.59570312, 0.59960938, 0.60351562,
            0.60742188, 0.61132812, 0.61523438, 0.61914062, 0.62304688,
            0.62695312, 0.63085938, 0.63476562, 0.63867188, 0.64257812,
            0.64648438, 0.65039062, 0.65429688, 0.65820312, 0.66210938,
            0.66601562, 0.66992188, 0.67382812, 0.67773438, 0.68164062,
            0.68554688, 0.68945312, 0.69335938, 0.69726562, 0.70117188,
            0.70507812, 0.70898438, 0.71289062, 0.71679688, 0.72070312,
            0.72460938, 0.72851562, 0.73242188, 0.73632812, 0.74023438,
            0.74414062, 0.74804688, 0.75195312, 0.75585938, 0.75976562,
            0.76367188, 0.76757812, 0.77148438, 0.77539062, 0.77929688,
            0.78320312, 0.78710938, 0.79101562, 0.79492188, 0.79882812,
            0.80273438, 0.80664062, 0.81054688, 0.81445312, 0.81835938,
            0.82226562, 0.82617188, 0.83007812, 0.83398438, 0.83789062,
            0.84179688, 0.84570312, 0.84960938, 0.85351562, 0.85742188,
            0.86132812, 0.86523438, 0.86914062, 0.87304688, 0.87695312,
            0.88085938, 0.88476562, 0.88867188, 0.89257812, 0.89648438,
            0.90039062, 0.90429688, 0.90820312, 0.91210938, 0.91601562,
            0.91992188, 0.92382812, 0.92773438, 0.93164062, 0.93554688,
            0.93945312, 0.94335938, 0.94726562, 0.95117188, 0.95507812,
            0.95898438, 0.96289062, 0.96679688, 0.97070312, 0.97460938,
            0.97851562, 0.98242188, 0.98632812, 0.99023438, 0.99414062,
            0.99804688])
        # fmt: on
    else:
        hard_coords = None

    # fmt: off
    start_points = np.array([[
        0.        , 1.        , 0.125     , 0.875     , 0.3515625 ,
        0.6484375 , 0.4296875 , 0.5703125 , 0.1875    , 0.8125    ,
        0.296875  , 0.703125  , 0.0703125 , 0.9296875 , 0.47265625,
        0.52734375, 0.390625  , 0.609375  , 0.22265625, 0.77734375,
        0.15625   , 0.84375
    ]])
    # fmt: on

    if n <= len(start_points):
        start = np.sort(start_points[2 : n + 1].flatten())
    else:
        start = mesh_interval_with_line2(n=2 * (n - 1))[0].get_coords()[1:-1:2, 0]

    def ref_mesh(coords):
        ref_nodes = XNodeSet()
        ref_nodes.add_node(np.array([0.0]))
        for coord in np.sort(coords):
            ref_nodes.add_node(np.array([coord]))
        ref_nodes.add_node(np.array([1.0]))
        ref_nodes.to_nodeset()

        ref_elems = XElementSet(ref_nodes)
        for ielem in np.arange(len(ref_nodes) - 1):
            inodes = np.array([ielem, ielem + 1])
            ref_elems.add_element(inodes)
        ref_elems.to_elementset()

        return ref_nodes, ref_elems

    def func(coords):
        ref_nodes, ref_elems = ref_mesh(coords)
        return -calc_norm(obs_elems, ref_elems)

    if hard_coords is None:
        print("optimizing {}-element mesh".format(len(start) + 1))

        lbounds = np.floor(start * 4) / 4
        ubounds = np.ceil(start * 4) / 4
        bounds = [(l, u) for l, u in zip(lbounds, ubounds)]
        result = minimize(func, start, bounds=bounds)

        print("optimal mesh found:")
        print(result.x)

        return ref_mesh(result.x)

    else:
        warn("Using hard-coded optimal coordinates")
        return ref_mesh(hard_coords)


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

    # posterior_norm = prior_norm_ref + prior_norm_obs
    # posterior_norm -= 2 * np.trace((K_ref.inv @ K_x @ K_obs.inv @ M_x.T).evaluate())

    posterior_norm = prior_norm_ref
    posterior_norm -= np.trace(
        (K_ref.inv @ K_x @ K_obs.inv @ K_x.T @ K_ref.inv @ M_ref).evaluate()
    )

    return posterior_norm

import os
import numpy as np
from scipy.sparse import save_npz, load_npz, issparse

from myjive.fem import Tri3Shape, DofSpace

from fem.jive import CJiveRunner
from fem.meshing import read_mesh, write_mesh, create_hypermesh

from experiments.inverse.frp_damage import params, misc, pod
from experiments.inverse.frp_damage.props import get_fem_props


def get_cache_folder():
    return os.path.join(os.path.dirname(__file__), "cache")


def get_cache_fname(name, dependencies):
    fname = name

    if "sparse" in dependencies:
        sparse = dependencies.pop("sparse")
    else:
        sparse = False

    for key in sorted(dependencies.keys()):
        value = dependencies[key]

        if key == "h":
            if isinstance(value, str):
                assert "r" in value or "d" in value or "h" in value
                fname += "_{}-{}".format(key, value)
            else:
                fname += "_{}-{:.3f}".format(key, value)
        else:
            fname += "_{}-{}".format(key, value)

    if sparse:
        fname += ".npz"
    else:
        fname += ".npy"

    return fname


def get_cache_fpath(name, dependencies):
    folder = get_cache_folder()
    fname = get_cache_fname(name, dependencies)
    return os.path.join(folder, fname)


def cache_exists():
    return os.path.isdir(get_cache_folder())


def is_cached(fpath):
    return os.path.isfile(fpath)


def write_cache(fpath, array):
    if not cache_exists():
        os.mkdir(get_cache_folder())

    ext = os.path.splitext(fpath)[1]

    if ext == ".npy":
        assert isinstance(array, np.ndarray)
        np.save(fpath, array)
    elif ext == ".npz":
        assert issparse(array)
        save_npz(fpath, array)
    else:
        assert False


def read_cache(fpath):
    ext = os.path.splitext(fpath)[1]

    if ext == ".npy":
        array = np.load(fpath)
        assert isinstance(array, np.ndarray)
        return array
    elif ext == ".npz":
        array = load_npz(fpath)
        assert issparse(array)
        return array
    else:
        assert False


def get_or_calc_fibers():
    n_fiber = params.geometry_params["n_fiber"]

    name = "fibers"
    dependencies = {"nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading fibers from cache")
        fibers = read_cache(path)
    else:
        print("Computing fibers")
        rve_size = params.geometry_params["rve_size"]
        r_fiber = params.geometry_params["r_fiber"]
        tol = params.geometry_params["tol_fiber"]
        seed = params.geometry_params["seed_fiber"]

        fibers = misc.calc_fibers(n=n_fiber, a=rve_size, r=r_fiber, tol=tol, seed=seed)
        print("Writing fibers to cache")
        write_cache(path, fibers)

    return fibers


def get_or_calc_mesh(*, h):
    n_fiber = params.geometry_params["n_fiber"]

    if isinstance(h, str):
        assert "r" in h
        fname = "meshes/rve_h-{}_nfib-{}.msh".format(h, n_fiber)
    else:
        fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)

    if not os.path.exists(fname):
        fibers = get_or_calc_fibers()

        print("Computing mesh")
        a = params.geometry_params["rve_size"]
        r = params.geometry_params["r_fiber"]
        misc.create_mesh(fibers=fibers, a=a, r=r, h=h, fname=fname, shift=False)

    print("Reading mesh from file")
    mesh = read_mesh(fname, read_groups=True)

    return mesh


def get_or_calc_dual_mesh(*, h):
    n_fiber = params.geometry_params["n_fiber"]
    assert "d" in h
    fname = "meshes/rve_h-{}_nfib-{}.msh".format(h, n_fiber)

    if not os.path.exists(fname):
        fibers = get_or_calc_fibers()

        print("Computing dual mesh")
        a = params.geometry_params["rve_size"]
        r = params.geometry_params["r_fiber"]
        misc.create_mesh(fibers=fibers, a=a, r=r, h=h, fname=fname, shift=True)

    print("Reading dual mesh from file")
    mesh = read_mesh(fname, read_groups=True)

    return mesh


def get_or_calc_hyper_mesh(*, h, do_groups=True):
    n_fiber = params.geometry_params["n_fiber"]
    assert "h" in h
    fname = "meshes/rve_h-{}_nfib-{}.msh".format(h, n_fiber)

    if os.path.exists(fname):
        print("Reading mesh from file")
        hyper_mesh = read_mesh(fname, read_groups=True)
    else:
        h_orig = float(h.split("h")[0])
        h_dual = h.replace("h", "d")
        orig_mesh = get_or_calc_mesh(h=h_orig)
        dual_mesh = get_or_calc_dual_mesh(h=h_dual)

        print("Computing hypermesh")
        hyper_mesh, _ = create_hypermesh(orig_mesh, dual_mesh, do_groups=do_groups)

        print("Writing hypermesh to cache")
        write_mesh(hyper_mesh, fname)

    return hyper_mesh


def get_or_calc_dofspace(*, h):
    n_fiber = params.geometry_params["n_fiber"]

    name = "dofspace"
    dependencies = {"nfib": n_fiber, "h": h}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading dofspace from cache")
        dofspace = read_cache(path)
    else:
        if isinstance(h, str):
            if "r" in h:
                mesh = get_or_calc_mesh(h=h)
            elif "d" in h:
                mesh = get_or_calc_dual_mesh(h=h)
            elif "h" in h:
                mesh = get_or_calc_hyper_mesh(h=h)
            else:
                assert False
        else:
            mesh = get_or_calc_mesh(h=h)

        nodes, elems, egroups = mesh

        print("Computing dofspace")
        props = get_fem_props()
        props["model"]["model"]["matrix"]["material"]["E"] = 1.0  # dummy value
        props["usermodules"]["solver"]["solver"] = {  # dummy solver
            "type": "GMRES",
            "precision": 1e100,
        }
        jive = CJiveRunner(props, elems=elems, egroups=egroups)
        globdat = jive()
        dofs = globdat["dofSpace"]

        ntype = dofs.type_count()
        nnode = dofs.dof_count() // ntype

        dofspace = np.zeros((nnode, ntype), dtype=int)

        for key in dofs.get_types():
            if key == "dx":
                j = 0
            elif key == "dy":
                j = 1
            else:
                assert False

            assert len(dofs._dofs[key]) == nnode

            inodes = list(dofs._dofs[key].keys())
            idofs = list(dofs._dofs[key].values())
            dofspace[inodes, j] = idofs

        print("Writing dofspace to cache")
        write_cache(path, dofspace)

    return dofspace


def get_or_calc_ipoints(*, egroup, h):
    n_fiber = params.geometry_params["n_fiber"]
    shape = Tri3Shape("Gauss3")

    name = "ipoints"
    dependencies = {"nfib": n_fiber, "h": h}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading ipoints from cache")
        ipoints = read_cache(path)
    else:
        print("Computing ipoints")
        ipoints = misc.calc_integration_points(egroup, shape)
        print("Writing ipoints to cache")
        write_cache(path, ipoints)

    return ipoints


def get_or_calc_distances(*, egroup, h):
    n_fiber = params.geometry_params["n_fiber"]

    name = "distances"
    dependencies = {"nfib": n_fiber, "h": h}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading distances from cache")
        distances = read_cache(path)
    else:
        fibers = get_or_calc_fibers()
        ipoints = get_or_calc_ipoints(egroup=egroup, h=h)
        r_fiber = params.geometry_params["r_fiber"]

        print("Computing distances")
        distances = np.zeros(ipoints.shape[0])
        for ip, ipoint in enumerate(ipoints):
            fiber, dist = misc.calc_closest_fiber(ipoint, fibers, 1.0)
            distances[ip] = dist - r_fiber

        if isinstance(h, str):
            if "r" in h:
                h = float(h.split("r")[0])
            elif "d" in h:
                h = float(h.split("d")[0])
            elif "h" in h:
                h = float(h.split("h")[0])
            else:
                assert False

        assert -0.1 * h < np.min(distances) < 0.2 * h
        print("Writing distances to cache")
        write_cache(path, distances)

    return distances


def get_or_calc_speckles():
    n_speckle = params.geometry_params["n_speckle"]

    name = "speckles"
    dependencies = {"nobs": n_speckle}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading speckles from cache")
        speckles = read_cache(path)
    else:
        print("Computing speckles")
        obs_size = params.geometry_params["obs_size"]
        r_speckle = params.geometry_params["r_speckle"]
        tol = params.geometry_params["tol_speckle"]
        seed = seed = params.geometry_params["seed_speckle"]

        speckles = misc.calc_fibers(
            n=n_speckle, a=obs_size, r=r_speckle, tol=tol, seed=seed
        )
        print("Writing speckles to cache")
        write_cache(path, speckles)

    return speckles


def get_or_calc_connectivity():
    n_speckle = params.geometry_params["n_speckle"]

    name = "connectivity"
    dependencies = {"nobs": n_speckle}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading connectivity from cache")
        connectivity = read_cache(path)
    else:
        speckles = get_or_calc_speckles()

        print("Computing connectivity")
        connectivity = misc.calc_connectivity(speckles)
        print("Writing connectivity to cache")
        write_cache(path, connectivity)

    return connectivity


def get_or_calc_dic_grid():
    h_dic = params.geometry_params["h_dic"]
    n_fiber = params.geometry_params["n_fiber"]

    name = "grid"
    dependencies = {"hdic": h_dic, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading grid from cache")
        grid = read_cache(path)
    else:
        fibers = get_or_calc_fibers()
        obs_size = params.geometry_params["obs_size"]
        rve_size = params.geometry_params["rve_size"]
        r_fiber = params.geometry_params["r_fiber"]

        print("Computing grid")
        grid = misc.calc_dic_grid(h_dic, fibers, r_fiber, obs_size, rve_size)
        print("Writing grid to cache")
        write_cache(path, grid)

    return grid


def get_or_calc_obs_operator(*, elems, h):
    n_fiber = params.geometry_params["n_fiber"]
    n_speckle = params.geometry_params["n_speckle"]
    shape = Tri3Shape("Gauss3")

    name = "observer"
    dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber, "sparse": True}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading observer from cache")
        obs_operator = read_cache(path)
    else:
        speckles = get_or_calc_speckles()
        connectivity = get_or_calc_connectivity()
        dofs = get_or_calc_dofspace(h=h)

        print("Computing observer")
        obs_operator = misc.calc_observer(speckles, connectivity, elems, dofs, shape)
        print("Writing observer to cache")
        write_cache(path, obs_operator)

    return obs_operator


def get_or_calc_dic_operator(*, elems, h):
    h_dic = params.geometry_params["h_dic"]
    n_fiber = params.geometry_params["n_fiber"]
    shape = Tri3Shape("Gauss3")

    name = "dicoperator"
    dependencies = {"hdic": h_dic, "h": h, "nfib": n_fiber, "sparse": True}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading dicoperator from cache")
        obs_operator = read_cache(path)
    else:
        fibers = get_or_calc_fibers()
        grid = get_or_calc_dic_grid()
        dofs = get_or_calc_dofspace(h=h)

        print("Computing dicoperator")
        obs_operator = misc.calc_dic_operator(fibers, grid, elems, dofs, shape)
        print("Writing dicoperator to cache")
        write_cache(path, obs_operator)

    return obs_operator


def get_or_calc_true_stiffnesses(*, egroup, h):
    n_fiber = params.geometry_params["n_fiber"]

    name = "stiffnesses"
    dependencies = {"h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading true stiffnesses from cache")
        stiffnesses = read_cache(path)
    else:
        E = params.material_params["E_matrix"]
        alpha = params.material_params["alpha"]
        beta = params.material_params["beta"]
        c = params.material_params["c"]
        d = params.material_params["d"]

        distances = get_or_calc_distances(egroup=egroup, h=h)
        stiffnesses = np.zeros(len(distances))

        print("Computing true stiffnesses")
        for ip, dist in enumerate(distances):
            sat = misc.saturation(dist, alpha, beta, c)
            dam = misc.damage(sat, d)
            stiffnesses[ip] = E * (1 - dam)

        print("Writing true stiffnesses to cache")
        write_cache(path, stiffnesses)

    return stiffnesses


def get_or_calc_true_displacements(*, egroups, h):
    n_fiber = params.geometry_params["n_fiber"]
    props = get_fem_props()

    name = "displacements"
    dependencies = {"h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading true displacements from cache")
        displacements = read_cache(path)
    else:
        egroup = egroups["matrix"]
        elems = egroup.get_elements()
        ipoints = get_or_calc_ipoints(egroup=egroup, h=h)
        stiffnesses = get_or_calc_true_stiffnesses(egroup=egroup, h=h)

        backdoor = {}
        backdoor["xcoord"] = ipoints[:, 0]
        backdoor["ycoord"] = ipoints[:, 1]

        stiffnesses = get_or_calc_true_stiffnesses(egroup=egroup, h=h)
        assert len(stiffnesses) == ipoints.shape[0]
        backdoor["e"] = stiffnesses

        print("Computing true displacements")
        jive = CJiveRunner(props, elems=elems, egroups=egroups)
        globdat = jive(**backdoor)
        displacements = globdat["state0"]

        print("Writing true displacements to cache")
        write_cache(path, displacements)

    return displacements


def get_or_calc_true_observations(*, h):
    n_fiber = params.geometry_params["n_fiber"]
    n_speckle = params.geometry_params["n_speckle"]

    name = "truth"
    dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading truth from cache")
        truth = read_cache(path)
    else:
        nodes, elems, egroups = get_or_calc_mesh(h=h)
        obs_operator = get_or_calc_obs_operator(elems=elems, h=h)
        true_displacements = get_or_calc_true_displacements(egroups=egroups, h=h)

        print("Computing truth")
        truth = obs_operator @ true_displacements
        print("Writing truth to cache")
        write_cache(path, truth)

    return truth


def get_or_calc_true_dic_observations(*, h):
    h_dic = params.geometry_params["h_dic"]
    n_fiber = params.geometry_params["n_fiber"]

    name = "dictruth"
    dependencies = {"hdic": h_dic, "h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading dictruth from cache")
        truth = read_cache(path)
    else:
        nodes, elems, egroups = get_or_calc_mesh(h=h)
        dic_operator = get_or_calc_dic_operator(elems=elems, h=h)
        true_displacements = get_or_calc_true_displacements(egroups=egroups, h=h)

        print("Computing dictruth")
        truth = dic_operator @ true_displacements
        print("Writing dictruth to cache")
        write_cache(path, truth)

    return truth


def get_or_calc_pod_snapshots(*, h):
    n_fiber = params.geometry_params["n_fiber"]
    n_snapshot = params.pod_params["n_snapshot"]

    name = "snapshots"
    dependencies = {"nsnap": n_snapshot, "h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading snapshots from cache")
        snapshots = read_cache(path)
    else:
        print("Computing snapshots")
        snapshots = pod.calc_snapshots(n_snapshot=n_snapshot, h=h)
        print("Writing snapshots to cache")
        write_cache(path, snapshots)

    return snapshots


def get_or_calc_pod_lifting(*, h):
    n_fiber = params.geometry_params["n_fiber"]

    name = "lifting"
    dependencies = {"h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading lifting from cache")
        lifting = read_cache(path)
    else:
        print("Computing lifting")
        lifting = pod.calc_lifting(h=h)
        print("Writing lifting to cache")
        write_cache(path, lifting)

    return lifting


def get_or_calc_pod_basis(*, h):
    n_fiber = params.geometry_params["n_fiber"]
    n_snapshot = params.pod_params["n_snapshot"]

    name = "basis"
    dependencies = {"nsnap": n_snapshot, "h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading basis from cache")
        basis = read_cache(path)
    else:
        snapshots = get_or_calc_pod_snapshots(h=h)
        lifting = get_or_calc_pod_lifting(h=h)

        for i in range(snapshots.shape[0]):
            snapshots[i] -= lifting

        print("Computing basis")
        basis, d, VT = np.linalg.svd(snapshots.T, full_matrices=False)

        print("Writing basis to cache")
        write_cache(path, basis)

    return basis

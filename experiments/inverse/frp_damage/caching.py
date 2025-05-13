import os
import numpy as np
from scipy.sparse import save_npz, load_npz, issparse

from myjive.fem import Tri3Shape, DofSpace

from fem.jive import CJiveRunner
from fem.meshing import read_mesh, write_mesh

from experiments.inverse.frp_damage import params, misc
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
    fname = "meshes/rve_h-{:.3f}_nfib-{}.msh".format(h, n_fiber)

    if not os.path.exists(fname):
        fibers = get_or_calc_fibers()

        print("Computing mesh")
        rve_size = params.geometry_params["rve_size"]
        r_fiber = params.geometry_params["r_fiber"]
        misc.create_mesh(fibers=fibers, a=rve_size, r=r_fiber, h=h, fname=fname)

    print("Reading mesh from file")
    mesh = read_mesh(fname, read_groups=True)

    return mesh


def get_or_calc_dofspace(*, h):
    n_fiber = params.geometry_params["n_fiber"]

    name = "dofspace"
    dependencies = {"nfib": n_fiber, "h": h}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading dofspace from cache")
        dofspace = read_cache(path)
    else:
        nodes, elems, egroups = get_or_calc_mesh(h=h)

        print("Computing dofspace")
        props = get_fem_props()
        props["model"]["model"]["matrix"]["material"]["E"] = 1.0  # dummy value
        jive = CJiveRunner(props, elems=elems, egroups=egroups)
        globdat = jive()
        dofs = globdat["dofSpace"]

        ntype = dofs.type_count()
        nnode = dofs.dof_count() // ntype

        dofspace = np.zeros((nnode, ntype))

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
        assert 0.0 < np.min(distances) < 0.1 * h
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


def get_or_calc_true_observations(*, egroups, h):
    n_fiber = params.geometry_params["n_fiber"]
    n_speckle = params.geometry_params["n_speckle"]

    name = "truth"
    dependencies = {"nobs": n_speckle, "h": h, "nfib": n_fiber}
    path = get_cache_fpath(name, dependencies)

    if is_cached(path):
        print("Reading truth from cache")
        truth = read_cache(path)
    else:
        elems = next(iter(egroups.values())).get_elements()
        obs_operator = get_or_calc_obs_operator(elems=elems, h=h)
        true_displacements = get_or_calc_true_displacements(egroups=egroups, h=h)

        print("Computing truth")
        truth = obs_operator @ true_displacements
        print("Writing truth to cache")
        write_cache(path, truth)

    return truth

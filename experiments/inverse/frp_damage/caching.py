import os
import numpy as np
from scipy.sparse import save_npz, load_npz, issparse


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

import os
import numpy as np


def get_cache_folder():
    return os.path.join(os.path.dirname(__file__), "cache")


def get_cache_fname(name, dependencies):
    fname = name
    for key in sorted(dependencies.keys()):
        value = dependencies[key]
        fname += "_" + str(key) + "-" + str(value)
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

    np.save(fpath, array)


def read_cache(fpath):
    return np.load(fpath)

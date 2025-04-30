import numpy as np

from fem.jive import CJiveRunner
from fem.meshing import read_mesh
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage.meshing import calc_closest_fiber

props = get_fem_props()
fname = props["userinput"]["gmsh"]["file"]
nodes, elems = read_mesh(fname)
jive = CJiveRunner(props, elems=elems)

globdat = jive()

from myjivex.util import QuickViewer, ElemViewer

QuickViewer(globdat["state0"], globdat, comp=0)

elems = globdat["elemSet"]
nodes = elems.get_nodes()

fibers = np.load("meshes/rve_nfib-16.npy")
distances = np.zeros(len(elems))

for ielem, inodes in enumerate(elems):
    coords = nodes[inodes]
    midpoint = np.mean(coords, axis=0)
    fiber, dist = calc_closest_fiber(midpoint, fibers, 1.0)
    distances[ielem] = max(dist - 0.1, 0.0)

ElemViewer(distances, globdat)

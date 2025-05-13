import numpy as np

from fem.jive import CJiveRunner
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

props = get_fem_props()

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h = 0.05

nodes, elems, egroups = caching.get_or_calc_mesh(h=h)
egroup = egroups["matrix"]

ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=egroup, h=h)

backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = ip_stiffnesses

props = get_fem_props()
jive = CJiveRunner(props, elems=elems, egroups=egroups)
globdat = jive(**backdoor)

eps_xx, eps_yy, gamma_xy = misc.calc_strains(globdat)
elem_stiffnesses = misc.calc_elem_stiffnesses(ip_stiffnesses, egroups)

from myjivex.util import QuickViewer, ElemViewer

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
)
ElemViewer(
    elem_stiffnesses,
    globdat,
    maxcolor=params.material_params["E_matrix"],
    title=r"stiffness, $N_e = {}$".format(len(elems)),
)

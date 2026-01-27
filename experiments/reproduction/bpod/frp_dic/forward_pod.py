import numpy as np

from myjive.solver import Constrainer

from fem.jive import CJiveRunner
from experiments.inverse.frp_damage.props import get_fem_props
from experiments.inverse.frp_damage import caching, misc, params

props = get_fem_props()

n_fiber = params.geometry_params["n_fiber"]
r_fiber = params.geometry_params["r_fiber"]
h = 0.02

nodes, elems, egroups = caching.get_or_calc_mesh(h=h)
egroup = egroups["matrix"]

ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)
ip_stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=egroup, h=h)
basis = caching.get_or_calc_pod_basis(h=h)
lifting = caching.get_or_calc_pod_lifting(h=h)

backdoor = {}
backdoor["xcoord"] = ipoints[:, 0]
backdoor["ycoord"] = ipoints[:, 1]
backdoor["e"] = ip_stiffnesses

props = get_fem_props()
jive = CJiveRunner(props, elems=elems, egroups=egroups)
globdat = jive(**backdoor)

K = globdat["matrix0"]
f = globdat["extForce"]
c = globdat["constraints"]
conman = Constrainer(c, K)
Kc = conman.get_output_matrix()
# fc = conman.get_rhs(f)

k = 20
Phi = basis[:, :k]

K_pod = Phi.T @ Kc @ Phi
f_pod = Phi.T @ f - Phi.T @ K @ lifting

u_pod = Phi @ np.linalg.solve(K_pod, f_pod) + lifting

from myjivex.util import QuickViewer, ElemViewer

QuickViewer(
    u_pod,
    globdat,
    comp=0,
)

QuickViewer(
    globdat["state0"],
    globdat,
    comp=0,
)

QuickViewer(
    u_pod - globdat["state0"],
    globdat,
    comp=0,
)

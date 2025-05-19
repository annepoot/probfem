import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from myjivex.util import QuickViewer

from fem.jive import CJiveRunner

from experiments.inverse.frp_damage import caching, misc, params, pod
from experiments.inverse.frp_damage.props import get_fem_props

for h in [0.05, 0.02, 0.01]:

    #################
    # get pod basis #
    #################

    snapshots = caching.get_or_calc_pod_snapshots(h=h)
    lifting = caching.get_or_calc_pod_lifting(h=h)
    basis = caching.get_or_calc_pod_basis(h=h)

    ###################
    # check pod basis #
    ###################

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

    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        phi_i = basis[:, i]
        cmax = np.max(np.abs(phi_i))
        cmin = -cmax
        QuickViewer(
            basis[:, i],
            globdat,
            comp=0,
            ax=ax1,
            mincolor=cmin,
            maxcolor=cmax,
            colorbar=None,
        )
        QuickViewer(
            basis[:, i],
            globdat,
            comp=1,
            ax=ax2,
            mincolor=cmin,
            maxcolor=cmax,
            colorbar=None,
        )
        plt.show()

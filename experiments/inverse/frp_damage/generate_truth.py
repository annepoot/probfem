import numpy as np

from experiments.inverse.frp_damage import caching, params

#######################
# get fiber locations #
#######################
fibers = caching.get_or_calc_fibers()

############
# get mesh #
############
h = 0.002
nodes, elems, egroups = caching.get_or_calc_mesh(h=h)

##########################
# get integration points #
##########################

egroup = egroups["matrix"]
ipoints = caching.get_or_calc_ipoints(egroup=egroup, h=h)

#######################
# get fiber distances #
#######################

distances = caching.get_or_calc_distances(egroup=egroup, h=h)

###################
# get stiffnesses #
###################

stiffnesses = caching.get_or_calc_true_stiffnesses(egroup=egroup, h=h)

#####################
# get displacements #
#####################

displacements = caching.get_or_calc_true_displacements(egroups=egroups, h=h)

################
# get dic grid #
################

grid = caching.get_or_calc_dic_grid()

############################
# get observation operator #
############################

dic_operator = caching.get_or_calc_dic_operator(elems=elems, h=h)

####################
# get ground truth #
####################

truth = caching.get_or_calc_true_dic_observations(h=h)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl

r_fiber = params.geometry_params["r_fiber"]
r_speckle = params.geometry_params["r_speckle"]
obs_size = params.geometry_params["obs_size"]

comp = 0
eps_xx = truth[comp::3]
norm = mpl.colors.Normalize(vmin=np.min(eps_xx), vmax=np.max(eps_xx))
cmap = mpl.colormaps["viridis"]

fig, ax = plt.subplots()

for fiber in fibers:
    ax.add_patch(Circle(fiber, r_fiber, color="C0", alpha=0.5))

for i, square in enumerate(grid):
    color = cmap(norm(eps_xx[i]))
    ax.add_patch(Rectangle(square[:2], square[2], square[3], color=color, alpha=0.5))

ax.set_aspect("equal")
ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
ax.set_axis_off()
plt.show()

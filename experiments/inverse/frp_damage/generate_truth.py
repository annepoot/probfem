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

#########################
# get speckle locations #
#########################

speckles = caching.get_or_calc_speckles()

#########################
# get speckle neighbors #
#########################

connectivity = caching.get_or_calc_connectivity()

############################
# get observation operator #
############################

obs_operator = caching.get_or_calc_obs_operator(elems=elems, h=h)

####################
# get ground truth #
####################

truth = caching.get_or_calc_true_observations(h=h)

import matplotlib.pyplot as plt
import matplotlib as mpl

comp = 0
edgedisp = truth[comp::2]
norm = mpl.colors.Normalize(vmin=np.min(edgedisp), vmax=np.max(edgedisp))
cmap = mpl.colormaps["viridis"]

fig, ax = plt.subplots()

for i, edge in enumerate(connectivity):
    coord1, coord2 = speckles[edge]
    dl = edgedisp[i]

    color = cmap(norm(dl))
    ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], color=color)

ax.set_aspect("equal")
obs_size = params.geometry_params["obs_size"]
ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
plt.show()

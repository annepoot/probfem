import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from experiments.inverse.frp_damage import caching, misc, params

#######################
# get fiber locations #
#######################
fibers = caching.get_or_calc_fibers()

#########################
# get speckle locations #
#########################

speckles = caching.get_or_calc_speckles()

#########################
# get speckle neighbors #
#########################

connectivity = caching.get_or_calc_connectivity()

r_fiber = params.geometry_params["r_fiber"]
r_speckle = params.geometry_params["r_speckle"]
obs_size = params.geometry_params["obs_size"]

fig, ax = plt.subplots()

for fiber in fibers:
    ax.add_patch(Circle(fiber, r_fiber, color="C0", alpha=0.5))

for speckle in speckles:
    ax.add_patch(Circle(speckle, 0.5 * r_speckle, color="C1", alpha=0.5))

for edge in connectivity:
    coords1, coords2 = speckles[edge]
    ax.plot([coords1[0], coords2[0]], [coords1[1], coords2[1]], color="C1", alpha=0.5)

ax.set_aspect("equal")

ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
plt.show()

############
# get mesh #
############

for h in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    nodes, elems, egroups = caching.get_or_calc_mesh(h=h)

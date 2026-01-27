import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from experiments.inverse.frp_damage import caching, params

#######################
# get fiber locations #
#######################
fibers = caching.get_or_calc_fibers()

################
# get dic grid #
################

grid = caching.get_or_calc_dic_grid()

r_fiber = params.geometry_params["r_fiber"]
r_speckle = params.geometry_params["r_speckle"]
obs_size = params.geometry_params["obs_size"]

fig, ax = plt.subplots()

for fiber in fibers:
    ax.add_patch(Circle(fiber, r_fiber, color="C0", alpha=0.5))

for square in grid:
    ax.add_patch(Rectangle(square[:2], square[2], square[3], color="0.5", alpha=0.5))

ax.set_aspect("equal")
ax.set_xlim((-obs_size, obs_size))
ax.set_ylim((-obs_size, obs_size))
ax.set_axis_off()
plt.show()

############
# get mesh #
############

for h in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    nodes, elems, egroups = caching.get_or_calc_mesh(h=h)

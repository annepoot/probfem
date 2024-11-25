import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
import myjive.util.proputils as pu
from bfem import declare_all as declarebfem
from myjivex import declare_all as declarex

props = pu.parse_file("tapered.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

x = np.linspace(0, 1, 65)
x_coarse = np.linspace(0, 1, 9)

# Get the prior and posterior means and standard deviations
u = globdat["ref"]["ref"]["state0"]
u_coarse = globdat["obs"]["obs"]["state0"]
# K = globdat["matrix0"].toarray()
# f = globdat["extForce"]

obsdat = globdat["obs"]["obs"]
refdat = globdat["ref"]["ref"]

u_prior = refdat["prior"]["mean"]
std_u_prior = refdat["prior"]["std"]
samples_u_prior = refdat["prior"]["samples"]
u_post = refdat["posterior"]["mean"]
std_u_post = refdat["posterior"]["std"]
samples_u_post = refdat["posterior"]["samples"]

plt.figure()
plt.plot(x, u_post, color="C0", label="posterior mean")
plt.plot(x, u_prior, color="C1", label="prior mean")
plt.plot(x, samples_u_post, color="C0", linewidth=0.2)
plt.plot(x, samples_u_prior, color="C1", linewidth=0.2)
plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
plt.fill_between(x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3)
plt.plot(x_coarse, u_coarse, color="C2", label="coarse solution")
plt.plot(x, u, color="C3", label="fine solution")
plt.ylim((-13, 17))
plt.legend(loc="upper left")
plt.show()

# plt.figure()
# plt.imshow(Kinv, vmax=11.5)
# plt.show()

# plt.figure()
# plt.imshow(Kinv - posterior.calc_cov(), vmax=11.5)
# plt.show()

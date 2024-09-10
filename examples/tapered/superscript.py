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

# Get the prior and posterior means and standard deviations
u = globdat["fine"]["tables"]["state0"]["dx"]
u_coarse = globdat["fine"]["tables"]["state0Coarse"]["dx"]

prior = globdat["gp"]["sequence"][3]
u_prior = prior.calc_mean()
std_u_prior = np.sqrt(np.diagonal(prior.calc_cov()))
samples_u_prior = prior.calc_samples(n=20, seed=0)

# Create figure 1 directly using matplotlib
for posterior in globdat["gp"]["sequence"]:
    u_post = posterior.calc_mean()
    std_u_post = np.sqrt(np.diagonal(posterior.calc_cov()))
    samples_u_post = posterior.calc_samples(n=20, seed=0)
    plt.figure()
    plt.plot(x, u_post, label="posterior mean")
    plt.plot(x, u_prior, label="prior mean")
    plt.plot(x, samples_u_post, color="C0", linewidth=0.2)
    plt.plot(x, samples_u_prior, color="C1", linewidth=0.2)
    plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
    plt.fill_between(x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3)
    plt.plot(x, u_coarse, color="C2", label="coarse solution")
    plt.plot(x, u, color="C3", label="fine solution")
    plt.ylim((-13, 17))
    plt.legend(loc="upper left")
    plt.show()

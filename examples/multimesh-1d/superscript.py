import numpy as np
import matplotlib.pyplot as plt

from myjive.app import main
import myjive.util.proputils as pu
from bfem import declare_all as declarebfem
from myjivex import declare_all as declarex
from probability import ConditionalGaussian

props = pu.parse_file("tapered.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

x = np.linspace(0, 1, 65)

# Get the prior and posterior means and standard deviations
u = globdat["tables"]["state0"]["dx"]
# u_coarse = globdat["tables"]["state0Coarse"]["dx"]
K = globdat["matrix0"].toarray()
f = globdat["extForce"]

K[0, :] = K[:, 0] = 0.0
K[-1, :] = K[:, -1] = 0.0
K[0, 0] = K[-1, -1] = 1.0
f[0] = f[-1] = 0.0

Kinv = np.zeros_like(K)
Kinv[1:-1, 1:-1] = np.linalg.inv(K[1:-1, 1:-1])

prior = globdat["gp"]["sequence"][0]
u_prior = prior.calc_mean()
std_u_prior = prior.calc_std()
samples_u_prior = prior.calc_samples(n=20, seed=0)

coarse_solves = []

n_test = 9
n_dof = len(u)
Phi_test = np.zeros((n_dof, n_test))
assert (n_dof - 1) % (n_test - 1) == 0
width = (n_dof - 1) // (n_test - 1)
for i in range(n_test - 1):
    Phi_test[i * width : (i + 1) * width + 1, i] = np.linspace(
        1, 0, width + 1, endpoint=True
    )
    Phi_test[i * width : (i + 1) * width + 1, i + 1] = np.linspace(
        0, 1, width + 1, endpoint=True
    )

# Create figure 1 directly using matplotlib
for posterior in globdat["gp"]["sequence"]:
    if isinstance(posterior._latent, ConditionalGaussian):
        linop = posterior._latent._linop
        if np.count_nonzero(linop) > 1:
            Kc = linop @ K @ linop.T + 1e-10 * np.identity(len(linop))
            fc = linop @ f

            uc = linop.T @ np.linalg.solve(Kc, fc)

            coarse_solves.append(uc)

    u_post = posterior.calc_mean()
    std_u_post = posterior.calc_std()
    samples_u_post = posterior.calc_samples(n=20, seed=0)
    plt.figure()
    plt.plot(x, u_post, color="C0", label="posterior mean")
    plt.plot(x, u_prior, color="C1", label="prior mean")
    plt.plot(x, samples_u_post, color="C0", linewidth=0.2)
    plt.plot(x, samples_u_prior, color="C1", linewidth=0.2)
    plt.fill_between(x, u_post - 2 * std_u_post, u_post + 2 * std_u_post, alpha=0.3)
    plt.fill_between(x, u_prior - 2 * std_u_prior, u_prior + 2 * std_u_prior, alpha=0.3)
    # plt.plot(x, u_coarse, color="C2", label="coarse solution")
    plt.plot(x, u, color="C3", label="fine solution")

    for uc in coarse_solves:
        plt.plot(x, uc, color="black", linewidth=0.5)

    plt.plot(x, Phi_test @ np.linalg.inv(Phi_test.T @ Phi_test) @ Phi_test.T @ u_post)
    plt.ylim((-13, 17))
    plt.legend(loc="upper left")
    plt.show()

plt.figure()
plt.imshow(Kinv, vmax=11.5)
plt.show()

plt.figure()
plt.imshow(Kinv - posterior.calc_cov(), vmax=11.5)
plt.show()

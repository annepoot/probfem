import numpy as np

from myjive.app import main
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem
from plate_props import props

props["model"]["bfem"]["prior"]["latent"]["cov"] = "K"

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["obs"]["obs"]
fglobdat = globdat

Phi = cglobdat["Phi"]
f = fglobdat["extForce"]

mean_u_post = globdat["gp"]["mean"]["posterior"]["state0"]
std_u_post = globdat["gp"]["std"]["posterior"]["state0"]
cov_u_post = globdat["gp"]["cov"]["posterior"]["state0"]

QuickViewer(
    mean_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior mean (Fig. 5a)",
    fname="img/K/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior standard deviation (Fig. 5b)",
    fname="img/K/std_state0-x_posterior.png",
)

pdNoise = 1e-4
cov_u_post += pdNoise**2 * np.identity(len(cov_u_post))

l, Q = np.linalg.eigh(cov_u_post)

newl = l * abs(Q.T @ f)
newcov = Q @ np.diag(newl) @ Q.T
newvar = newcov.diagonal()
newstd = np.sqrt(newvar)

QuickViewer(
    cov_u_post @ f,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Error recovery (Fig. 5c)",
    fname="img/K/std_state0-x_error_recovered.png",
)
QuickViewer(
    newstd,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Rescaled posterior standard deviation (Fig. 5d)",
    fname="img/K/std_state0-x_posterior_rescaled.png",
)

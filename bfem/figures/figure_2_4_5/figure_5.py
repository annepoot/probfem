import numpy as np

from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")
props["model"]["bfem"]["prior"]["latent"]["cov"] = "K"

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["obs"]["obs"]
fglobdat = globdat

u_coarse = cglobdat["state0"]
u = fglobdat["state0"]
f = fglobdat["extForce"]

mean = globdat["gp"]["mean"]
u_post = mean["posterior"]["state0"]

std = globdat["gp"]["std"]
std_u_post = std["posterior"]["state0"]

cov = globdat["gp"]["cov"]
cov_u_post = cov["posterior"]["state0"]

Phi = cglobdat["Phi"]

QuickViewer(
    u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/K/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/K/std_state0-x_posterior.png",
)

dc = len(u)
pdNoise = 1e-4

cov_u_post += pdNoise**2 * np.identity(dc)

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
    fname="img/K/std_state0-x_error_recovered.png",
)
QuickViewer(
    newstd,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/K/std_state0-x_posterior_rescaled.png",
)

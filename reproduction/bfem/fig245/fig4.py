from myjive.app import main
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem
from plate_props import props

props["model"]["bfem"]["prior"]["latent"]["cov"] = "M"

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["obs"]["obs"]
fglobdat = globdat

Phi = cglobdat["Phi"]

mean_u_post = globdat["gp"]["mean"]["posterior"]["state0"]
std_u_post = globdat["gp"]["std"]["posterior"]["state0"]

QuickViewer(
    mean_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior mean (Fig. 4a)",
    fname="img/M/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Posterior standard deviation (Fig. 4b)",
    fname="img/M/std_state0-x_posterior.png",
)

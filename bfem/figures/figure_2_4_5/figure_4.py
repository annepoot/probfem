from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")
props["model"]["bfem"]["prior"]["latent"]["cov"] = "M"

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["coarse"]
fglobdat = globdat["fine"]

u_coarse = cglobdat["state0"]
u = fglobdat["state0"]

mean = globdat["gp"]["mean"]
u_post = mean["posterior"]["state0"]

std = globdat["gp"]["std"]
std_u_post = std["posterior"]["state0"]

Phi = globdat["Phi"]

QuickViewer(
    u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/M/mean_state0-x_posterior.png",
)
QuickViewer(
    std_u_post,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/M/std_state0-x_posterior.png",
)

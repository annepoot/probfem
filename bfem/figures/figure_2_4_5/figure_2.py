from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["coarse"]
fglobdat = globdat["fine"]

u_coarse = cglobdat["state0"]
eps_xx_coarse = cglobdat["tables"]["strain"]["xx"]
u = fglobdat["state0"]
eps_xx = fglobdat["tables"]["strain"]["xx"]

Phi = globdat["Phi"]

err = u - Phi @ u_coarse
err_eps_xx = eps_xx - Phi[: len(eps_xx), : len(eps_xx_coarse)] @ eps_xx_coarse

QuickViewer(
    u_coarse,
    cglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/state0-x_coarse.png",
)
QuickViewer(
    u,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/state0-x.png",
)
QuickViewer(
    err,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/error_state0-x.png",
)

QuickViewer(
    eps_xx_coarse,
    cglobdat,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/strain-xx_coarse.png",
)
QuickViewer(
    eps_xx,
    fglobdat,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/strain-xx.png",
)
QuickViewer(
    err_eps_xx,
    fglobdat,
    dpi=600,
    figsize=(7.5, 3),
    fname="img/core-plots/error_strain-xx.png",
)

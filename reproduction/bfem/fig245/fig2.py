from myjive.app import main
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem
from plate_props import props

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)
cglobdat = globdat["obs"]["obs"]
fglobdat = globdat

u_coarse = cglobdat["state0"]
u = fglobdat["state0"]
Phi = cglobdat["Phi"]
err = u - Phi @ u_coarse

QuickViewer(
    err,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Discretization error (Fig. 2b)",
    fname="img/core-plots/error_state0-x.png",
)
QuickViewer(
    u,
    fglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Fine-scale solution (Fig. 2c)",
    fname="img/core-plots/state0-x.png",
)
QuickViewer(
    u_coarse,
    cglobdat,
    comp=0,
    dpi=600,
    figsize=(7.5, 3),
    title="Coarse-scale solution (Fig. 2d)",
    fname="img/core-plots/state0-x_coarse.png",
)

from myjivex.util import QuickViewer
from fem import JiveRunner
from fem_props import get_fem_props
from meshing import create_phi_from_globdat

coarse_props = get_fem_props("meshes/plate_r0.msh")
fine_props = get_fem_props("meshes/plate_r1.msh")

cjive = JiveRunner(coarse_props)
cglobdat = cjive()
fjive = JiveRunner(fine_props)
fglobdat = fjive()

Phi = create_phi_from_globdat(cglobdat, fglobdat)
u_coarse = cglobdat["state0"]
u = fglobdat["state0"]
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

from myjivex.util import QuickViewer
from fem.jive import CJiveRunner
from fem.meshing import create_phi_from_globdat, read_mesh
from experiments.reproduction.bfem.fig245.fem_props import get_fem_props

_, coarse_elems = read_mesh("meshes/plate_r0.msh")
_, fine_elems = read_mesh("meshes/plate_r1.msh")

props = get_fem_props()

cjive = CJiveRunner(props, elems=coarse_elems)
fjive = CJiveRunner(props, elems=fine_elems)

cglobdat = cjive()
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

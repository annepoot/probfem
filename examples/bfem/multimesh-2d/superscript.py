from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("box.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

prior = globdat["gp"]["sequence"][0]
u_prior = prior.calc_mean()
std_u_prior = prior.calc_std()
posterior = globdat["gp"]["sequence"][-1]
u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

cglobdat = globdat["obs"]["obs0"]

for i in range(10):
    cglobdat = globdat["obs"]["obs{}".format(i)]
    linop = globdat["obs"]["obs"]["Phi"].T[18 * i : 18 * (i + 1)]
    u_coarse = cglobdat["state0"]
    QuickViewer(
        cglobdat["state0"],
        cglobdat,
        comp=0,
        title="Coarse solution {}".format(i),
        linewidth=0.2,
    )
    QuickViewer(
        linop.T @ u_coarse,
        globdat,
        comp=0,
        title="Coarse solution {} (projected)".format(i),
        linewidth=0.2,
    )

QuickViewer(globdat["state0"], globdat, comp=0, title="Fine solution", linewidth=0.2)
QuickViewer(u_post, globdat, comp=0, title="Posterior mean", linewidth=0.2)
QuickViewer(std_u_post, globdat, comp=0, title="Posterior std", linewidth=0.2)

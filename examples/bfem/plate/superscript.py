from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

prior = globdat["gp"]["sequence"][0]
u_prior = prior.calc_mean()
std_u_prior = prior.calc_std()
posterior = globdat["gp"]["sequence"][-1]
u_post = posterior.calc_mean()
std_u_post = posterior.calc_std()

cglobdat = globdat["obs"]["obs"]

QuickViewer(cglobdat["state0"], cglobdat, comp=0, title="Coarse solution")
QuickViewer(u_post, globdat, comp=0, title="Posterior mean")
QuickViewer(std_u_post, globdat, comp=0, title="Posterior std")

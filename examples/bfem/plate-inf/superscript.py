from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

# Get the prior and posterior means and standard deviations
u = globdat["ref"]["ref"]["state0"]
u_coarse = globdat["obs"]["obs"]["state0"]
# K = globdat["matrix0"].toarray()
# f = globdat["extForce"]

obsdat = globdat["obs"]["obs"]
refdat = globdat["ref"]["ref"]

u_prior = refdat["prior"]["mean"]
std_u_prior = refdat["prior"]["std"]
samples_u_prior = refdat["prior"]["samples"]
u_post = refdat["posterior"]["mean"]
std_u_post = refdat["posterior"]["std"]
cov_u_post = refdat["posterior"]["cov"]
samples_u_post = refdat["posterior"]["samples"]

QuickViewer(u, refdat, comp=0, title="Reference solution")
QuickViewer(u_coarse, obsdat, comp=0, title="Coarse solution")
QuickViewer(u_post, refdat, comp=0, title="Posterior mean")
QuickViewer(std_u_post, refdat, comp=0, title="Posterior std")

error_est = cov_u_post @ refdat["extForce"]

QuickViewer(error_est, refdat, comp=0, title="Posterior std")

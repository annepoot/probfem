from myjive.app import main
import myjive.util.proputils as pu
from rmfem import declare_all as declarermfem
from myjivex import declare_all as declarex
from myjivex.modules import ViewModule

extra_declares = [declarex, declarermfem]
props = pu.parse_file("2d-rmfem-for.pro")

globdat = main.jive(props, extra_declares=extra_declares)

for pglobdat in globdat["perturbedSolves"]:
    view = ViewModule("view")

    viewprops = {
        # "type": "View",
        "plotType": "node",
        "tables": ["stiffness", "solution"],
        "comps": ["", "dx"],
        "fill": {"levels": 100},
        "line": {"linewidth": 1},
        "colorbar": {"show": True},
    }
    props["view"] = viewprops

    view.configure(pglobdat, **viewprops)
    view.shutdown(pglobdat)

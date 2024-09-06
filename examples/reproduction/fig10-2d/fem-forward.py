from myjive.app import main
import myjive.util.proputils as pu
from myjivex import declare_all as declarex

extra_declares = [declarex]
props = pu.parse_file("2d-fem-for.pro")

outputprops = {
    "type": "Output",
    "files": ["output/state0.csv", "output/stiffness.csv"],
    "keys": ["state0", "tables.stiffness"],
    "overwrite": True,
}
props["output"] = outputprops

viewprops = {
    "type": "View",
    "plotType": "node",
    "tables": ["stiffness", "solution"],
    "comps": ["", "dx"],
    "fill": {"levels": 100},
    "colorbar": {"show": True},
}
props["view"] = viewprops

globdat = main.jive(props, extra_declares=extra_declares)

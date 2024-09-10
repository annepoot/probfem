from myjive.app import main
import myjive.util.proputils as pu
from myjivex.util import QuickViewer
from myjivex import declare_all as declarex
from bfem import declare_all as declarebfem

props = pu.parse_file("plate.pro")

extra_declares = [declarex, declarebfem]
globdat = main.jive(props, extra_declares=extra_declares)

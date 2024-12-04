from myjive.app import main
from myjivex.declare import declare_all

__all__ = ["JiveRunner"]


class JiveRunner:

    def __init__(self, props):
        self.props = props

    def __call__(self):
        globdat = main.jive(self.props, extra_declares=[declare_all])
        return globdat

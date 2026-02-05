def get_fem_props(dimensionality):

    if dimensionality == 1:
        return get_fem_props_1d()
    elif dimensionality == 2:
        return get_fem_props_2d()
    else:
        assert False


def get_fem_props_1d():
    fem_props_1d = {
        "log": {
            "pattern": "*.info",
            "rank": 1,
            "file": "-$(CASE_NAME).log",
        },
        "control": {"runWhile": "i<1"},
        "shapeTable": {
            "type": "Auto",
            "boundaryElems": "",
            "interfaceElems": "",
            "maxPrecision": 4,
        },
        "userinput": {
            "modules": ["globdat", "ngroups"],
            "globdat": {"type": "GlobdatInput"},
            "ngroups": {
                "type": "GroupInput",
                "nodeGroups": ["left", "right"],
                "left.xtype": "min",
                "right.xtype": "max",
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["laplace", "diri", "source"],
                "laplace": {
                    "type": "Laplace",
                    "elements": "all",
                    "shape": {"type": "Line2", "intScheme": "Gauss4"},
                },
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": 0.0,
                    "dispIncr": 0.0,
                    "nodeGroups": ["left", "right"],
                    "dofs": ["u", "u"],
                    "factors": [0.0, 0.0],
                },
                "source": {
                    "type": "Source",
                    "elements": "all",
                    "sourceFunc": "-(2 + exp(2 * x) * (2 - 8*x - 4*x*x))",
                    "precision": 4,
                },
            },
        },
        "usermodules": {
            "modules": ["solver"],
            "solver": {
                "type": "Linsolve",
            },
        },
    }
    return fem_props_1d


def get_fem_props_2d():
    fem_props_2d = {
        "log": {
            "pattern": "*.info",
            "rank": 1,
            "file": "-$(CASE_NAME).log",
        },
        "control": {"runWhile": "i<1"},
        "shapeTable": {
            "type": "Auto",
            "boundaryElems": "",
            "interfaceElems": "",
            "maxPrecision": 4,
        },
        "userinput": {
            "modules": ["globdat", "ngroups"],
            "globdat": {"type": "GlobdatInput"},
            "ngroups": {
                "type": "GroupInput",
                "nodeGroups": ["left", "right", "bottom", "top"],
                "left.xtype": "min",
                "right.xtype": "max",
                "bottom.ytype": "min",
                "top.ytype": "max",
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["laplace", "diri", "source"],
                "laplace": {
                    "type": "Laplace",
                    "elements": "all",
                    "shape": {"type": "Triangle3", "intScheme": "Gauss3"},
                },
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": 0.0,
                    "dispIncr": 0.0,
                    "nodeGroups": ["left", "right", "bottom", "top"],
                    "dofs": ["u", "u", "u", "u"],
                    "factors": [0.0, 0.0, 0.0, 0.0],
                },
                "source": {
                    "type": "Source",
                    "elements": "all",
                    # u = ux(x) * uy(y)
                    # source = -(d^ux/dx^2 * uy + ux * d^uy/dy^2)
                    "sourceFunc": "-((2 + exp(2 * x) * (2 - 8*x - 4*x*x)) * sin(3.14159265358979323846 * y) + (1 - x*x) * (exp(2 * x) - 1) * -3.14159265358979323846*3.14159265358979323846 * sin(3.14159265358979323846 * y))",
                    "precision": 4,
                },
            },
        },
        "usermodules": {
            "modules": ["solver"],
            "solver": {
                "type": "Linsolve",
            },
        },
    }
    return fem_props_2d

__all__ = ["get_fem_props"]


def get_fem_props():
    fem_props = {
        "log": {
            "pattern": "*.info",
            "rank": 1,
            "file": "-$(CASE_NAME).log",
        },
        "control": {
            "runWhile": "i<1",
        },
        "shapeTable": {
            "type": "Auto",
            "boundaryElems": "",
            "interfaceElems": "",
            "maxPrecision": 4,
        },
        "userinput": {
            "modules": ["globdat", "ngroups"],
            "globdat": {
                "type": "GlobdatInput",
            },
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
                "models": ["elastic", "load", "diri"],
                "elastic": {
                    "type": "Elastic",
                    "elements": "all",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "BAR",
                        "rank": 1,
                        "E": "exp(sqrt(2) * (xi1/(pi) * sin(pi*x) + xi2/(2*pi) * sin(2*pi*x) + xi3/(3*pi) * sin(3*pi*x) + xi4/(4*pi) * sin(4*pi*x)))",
                        "area": 1.0,
                        "params": {
                            "names": ["xi1", "xi2", "xi3", "xi4", "pi"],
                            "values": [1.0, 1.0, 0.25, 0.25, 3.14159265358979323846],
                        },
                    },
                    "shape": {
                        "type": "Line2",
                        "intScheme": "Gauss4",
                    },
                },
                "load": {
                    "type": "Load",
                    "elements": "all",
                    "dofs": ["dx"],
                    "load": ["sin(2 * 3.14159265358979323846 * x)"],
                    "precision": 4,
                },
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": 0.0,
                    "dispIncr": 0.0,
                    "nodeGroups": ["left", "right"],
                    "dofs": ["dx", "dx"],
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
    return fem_props

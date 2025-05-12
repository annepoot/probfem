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
        "userinput": {
            "modules": ["globdat", "ngroups"],
            "globdat": {
                "type": "GlobdatInput",
            },
            "ngroups": {
                "type": "GroupInput",
                "nodeGroups": ["l", "r", "b", "t", "lb", "rb", "lt", "rt"],
                "l": {
                    "xtype": "min",
                },
                "r": {
                    "xtype": "max",
                },
                "b": {
                    "ytype": "min",
                },
                "t": {
                    "ytype": "max",
                },
                "lb": {
                    "xtype": "min",
                    "ytype": "min",
                },
                "rb": {
                    "xtype": "max",
                    "ytype": "min",
                },
                "lt": {
                    "xtype": "min",
                    "ytype": "max",
                },
                "rt": {
                    "xtype": "max",
                    "ytype": "max",
                },
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["matrix", "fiber", "diri"],
                "matrix": {
                    "type": "Elastic",
                    "elements": "matrix",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": "backdoor",
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
                },
                "fiber": {
                    "type": "Elastic",
                    "elements": "fiber",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": 22000.0,
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
                },
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": -0.01,
                    "dispIncr": 0.0,
                    "nodeGroups": ["l", "l", "r", "r"],
                    "dofs": ["dx", "dy", "dx", "dy"],
                    "factors": [0.0, 0.0, 1.0, 0.0],
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

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
                "nodeGroups": ["left", "topright", "botleft"],
                "left.xtype": "min",
                "botleft.xtype": "min",
                "botleft.ytype": "min",
                "topright.xtype": "max",
                "topright.ytype": "max",
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["elastic", "diri"],
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": -0.01,
                    "dispIncr": 0.0,
                    "nodeGroups": ["left", "botleft", "topright"],
                    "dofs": ["dx", "dy", "dy"],
                    "loaded": 2,
                },
                "elastic": {
                    "type": "Elastic",
                    "elements": "all",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": 1000.0,
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
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

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
            "modules": ["gmsh", "ngroups"],
            "gmsh": {
                "type": "GmshInput",
                "dim": 2,
                "file": "meshes/rve_h-0.05_nfib-30.msh",
                "doElemGroups": True,
            },
            "ngroups": {
                "type": "GroupInput",
                "nodeGroups": ["left", "right", "bottom", "top"],
                "left": {
                    "xtype": "min",
                },
                "right": {
                    "xtype": "max",
                },
                "bottom": {
                    "ytype": "min",
                },
                "top": {
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
                        "E": 30000.0,
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
                    "nodeGroups": ["left", "right", "bottom", "top"],
                    "dofs": ["dx", "dx", "dy", "dy"],
                    "factors": [0.0, 1.0, 0.0, 0.0],
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

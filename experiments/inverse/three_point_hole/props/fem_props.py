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
                "nodeGroups": ["leftpoint", "midpoint", "rightpoint"],
                "leftpoint": {
                    "x0": 0.5,
                    "y0": -0.1,
                    "radius": 1e-8,
                },
                "midpoint": {
                    "x0": 2.5,
                    "y0": 1.1,
                    "radius": 1e-8,
                },
                "rightpoint": {
                    "x0": 4.5,
                    "y0": -0.1,
                    "radius": 1e-8,
                },
                "elemGroups": ["beam", "leftrect", "midrect", "rightrect"],
                "beam": {
                    "xbounds": [0.0, 5.0],
                    "ybounds": [0.0, 1.0],
                },
                "leftrect": {
                    "xbounds": [4.4, 4.6],
                    "ybounds": [-0.1, 0.0],
                },
                "midrect": {
                    "xbounds": [2.4, 2.6],
                    "ybounds": [1.0, 1.1],
                },
                "rightrect": {
                    "xbounds": [0.4, 0.6],
                    "ybounds": [-0.1, 0.0],
                },
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["elastic", "leftsup", "midsup", "rightsup", "diri"],
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": -0.01,
                    "dispIncr": 0.0,
                    "nodeGroups": ["leftpoint", "leftpoint", "midpoint", "rightpoint"],
                    "dofs": ["dx", "dy", "dy", "dy"],
                    "factors": [0.0, 0.0, 1.0, 0.0],
                },
                "elastic": {
                    "type": "Elastic",
                    "elements": "beam",
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
                "leftsup": {
                    "type": "Elastic",
                    "elements": "leftrect",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": 1e9,
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
                },
                "midsup": {
                    "type": "Elastic",
                    "elements": "midrect",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": 1e9,
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
                },
                "rightsup": {
                    "type": "Elastic",
                    "elements": "rightrect",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRAIN",
                        "rank": 2,
                        "E": 1e9,
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

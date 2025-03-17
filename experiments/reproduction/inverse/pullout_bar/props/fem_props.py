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
                "models": ["elastic", "spring", "neum"],
                "elastic": {
                    "type": "Elastic",
                    "elements": "all",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "BAR",
                        "rank": 1,
                        "E": 1.2,
                        "area": 1.0,
                    },
                    "shape": {
                        "type": "Line2",
                        "intScheme": "Gauss4",
                    },
                },
                "spring": {
                    "type": "Spring",
                    "elements": "all",
                    "k": 110.0,
                    "shape": {
                        "type": "Line2",
                        "intScheme": "Gauss4",
                    },
                },
                "neum": {
                    "type": "Neumann",
                    "initLoad": 10.0,
                    "loadIncr": 0.0,
                    "nodeGroups": ["right"],
                    "dofs": ["dx"],
                    "factors": [1.0],
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

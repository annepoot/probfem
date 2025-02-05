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
                        "E": "0.1 - 0.099 * x",
                        "area": 1.0,
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
                    "load": [3.0],
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

def get_fem_props():
    fem_props = {
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
                "models": ["elastic", "diri", "load"],
                "elastic": {
                    "type": "Elastic",
                    "elements": "all",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "BAR",
                        "rank": 1,
                        "E": 1.0,
                        "area": 1.0,
                    },
                    "shape": {"type": "Line2", "intScheme": "Gauss4"},
                },
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": 0.0,
                    "dispIncr": 0.0,
                    "nodeGroups": ["left", "right"],
                    "dofs": ["dx", "dx"],
                    "factors": [0.0, 0.0],
                },
                "load": {
                    "type": "Load",
                    "elements": "all",
                    "dofs": ["dx"],
                    "load": ["-2 - exp(2 * x) * (2 - 8*x - 4*x*x)"],
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
    return fem_props

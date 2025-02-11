def get_fem_props():
    fem_props = {
        "log": {
            "pattern": "*.info",
            "rank": 1,  # print warnings, but not log or out
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
                "nodeGroups": ["l", "lb"],
                "l.xtype": "min",
                "lb.xtype": "min",
                "lb.ytype": "min",
            },
        },
        "model": {
            "type": "Matrix",
            "model": {
                "type": "Multi",
                "models": ["elastic", "diri", "load"],
                "diri": {
                    "type": "Dirichlet",
                    "initDisp": 1.0,
                    "dispIncr": 0.0,
                    "nodeGroups": ["l", "lb"],
                    "dofs": ["dx", "dy"],
                },
                "elastic": {
                    "type": "Elastic",
                    "elements": "all",
                    "material": {
                        "type": "LinearIsotropic",
                        "anmodel": "PLANE_STRESS",
                        "rank": 2,
                        "E": 3.0,
                        "nu": 0.2,
                    },
                    "shape": {
                        "type": "Triangle3",
                        "intScheme": "Gauss3",
                    },
                },
                "load": {
                    "type": "Load",
                    "elements": "all",
                    "dofs": ["dx"],
                    "load": [1.0],
                },
            },
        },
        "usermodules": {
            "modules": ["solver"],
            "solver": {"type": "Linsolve"},
        },
    }
    return fem_props

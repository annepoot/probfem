def get_fem_props(fname):
    fem_props = {
        "modules": ["init", "solver"],
        "init": {
            "type": "Init",
            "mesh": {
                "type": "gmsh",
                "file": fname,
            },
            "nodeGroups": ["l", "r", "b", "t"],
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
        },
        "solve": {
            "type": "Linsolve",
        },
        "model": {
            "models": ["solid", "load", "diri"],
            "solid": {
                "type": "Solid",
                "elements": "all",
                "material": {
                    "type": "Isotropic",
                    "rank": 2,
                    "anmodel": "plane_stress",
                    "E": 3.0,
                    "nu": 0.2,
                },
                "shape": {
                    "type": "Quad4",
                    "intScheme": "Gauss9",
                },
            },
            "load": {
                "type": "Load",
                "elements": "all",
                "dofs": ["dx"],
                "values": [1.0],
                "shape": {
                    "type": "Quad4",
                    "intScheme": "Gauss9",
                },
            },
            "diri": {
                "type": "Dirichlet",
                "groups": ["l", "r", "b", "t", "l", "r", "b", "t"],
                "dofs": ["dx", "dx", "dx", "dx", "dy", "dy", "dy", "dy"],
                "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }
    return fem_props

def get_fem_props(fname):
    fem_props = {
        "modules": ["init", "solve"],
        "init": {
            "type": "Init",
            "mesh": {"type": "gmsh", "file": fname},
            "nodeGroups": ["l", "lb"],
            "l": {"xtype": "min"},
            "lb": {"xtype": "min", "ytype": "min"},
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
                "shape": {"type": "Triangle3", "intScheme": "Gauss1"},
            },
            "load": {
                "type": "Load",
                "elements": "all",
                "dofs": ["dx"],
                "values": [1.0],
                "shape": {"type": "Triangle3", "intScheme": "Gauss1"},
            },
            "diri": {
                "type": "Dirichlet",
                "groups": ["l", "lb"],
                "dofs": ["dx", "dy"],
                "values": [0.0, 0.0],
            },
        },
    }
    return fem_props

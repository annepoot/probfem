def get_fem_props(fname):
    fem_props = {
        "modules": ["init", "solve"],
        "init": {
            "type": "Init",
            "mesh": {"type": "manual", "file": fname},
            "nodeGroups": ["left", "right"],
            "left": {"xtype": "min"},
            "right": {"xtype": "max"},
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
                    "type": "Heterogeneous",
                    "rank": 1,
                    "anmodel": "bar",
                    "E": "0.1 - 0.099 * x",
                },
                "shape": {"type": "Line2", "intScheme": "Gauss2"},
            },
            "load": {
                "type": "Load",
                "elements": "all",
                "dofs": ["dx"],
                "values": [3.0],
                "shape": {"type": "Line2", "intScheme": "Gauss2"},
            },
            "diri": {
                "type": "Dirichlet",
                "groups": ["left", "right"],
                "dofs": ["dx", "dx"],
                "values": [0.0, 0.0],
            },
        },
    }
    return fem_props

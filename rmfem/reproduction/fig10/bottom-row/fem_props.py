fem_props = {
    "modules": ["init", "solve"],
    "init": {
        "type": "Init",
        "mesh": {"type": "manual", "file": "1d-lin.mesh"},
        "nodeGroups": ["left", "right"],
        "left": {"xtype": "min"},
        "right": {"xtype": "max"},
    },
    "solve": {
        "type": "Linsolve",
        "tables": ["stiffness"],
        "elemTables": ["strain", "size"],
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
                "E": "exp(sqrt(2) * (xi_1/(pi) * sin(pi*x) + xi_2/(2*pi) * sin(2*pi*x) + xi_3/(3*pi) * sin(3*pi*x) + xi_4/(4*pi) * sin(4*pi*x)))",
                "params": {
                    "xi_1": 1.0,
                    "xi_2": 1.0,
                    "xi_3": 0.25,
                    "xi_4": 0.25,
                },
            },
            "shape": {"type": "Line2", "intScheme": "Gauss4"},
        },
        "load": {
            "type": "Load",
            "elements": "all",
            "dofs": ["dx"],
            "values": ["sin(2*pi*x)"],
            "shape": {"type": "Line2", "intScheme": "Gauss4"},
        },
        "diri": {
            "type": "Dirichlet",
            "groups": ["left", "right"],
            "dofs": ["dx", "dx"],
            "values": [0.0, 0.0],
        },
    },
}

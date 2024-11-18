from myjive.app import main

fig2_props = {
    "inner": {
        "type": main.jive,
        "modules": ["init", "solve"],
        "init": {
            "type": "Init",
            "mesh": {"type": "manual", "file": "fig2.mesh"},
            "nodeGroups": ["left", "right"],
            "left": {"xtype": "min"},
            "right": {"xtype": "max"},
        },
        "solve": {
            "type": "Linsolve",
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
                    "E": 1,
                },
                "shape": {"type": "Line2", "intScheme": "Gauss1"},
            },
            "load": {
                "type": "Load",
                "elements": "all",
                "dofs": ["dx"],
                # forcing term yields u(x) = sin(2*pi*x)
                # if kappa = 1
                "values": ["4*pi**2 * sin(2*pi*x)"],
                "shape": {"type": "Line2", "intScheme": "Gauss2"},
            },
            "diri": {
                "type": "Dirichlet",
                "groups": ["left", "right"],
                "dofs": ["dx", "dx"],
                "values": [0.0, 0.0],
            },
        },
    },
    "p": 1,
    "n_sample": 50,
    "seed": 0,
    "update_type": "in_place",
    "run_modules": ["solve"],
    "globdat_keys": ["state0", "elemSet", "nodeSet"],
}

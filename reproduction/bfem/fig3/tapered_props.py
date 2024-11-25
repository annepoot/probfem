props = {
    "modules": ["init", "solve"],
    "init": {
        "type": "Init",
        "mesh": {"type": "manual", "file": "bar_fine.mesh"},
        "nodeGroups": ["left", "right"],
        "left": {"xtype": "min"},
        "right": {"xtype": "max"},
    },
    "solve": {
        "type": "BFEMSolve",
        "fineSolve": {
            "type": "Linsolve",
        },
        "sequential": False,
        "nsample": 20,
    },
    "model": {
        "models": ["solid", "bfem", "load", "diri", "bobs", "obs"],
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
        "bfem": {
            "type": "BFEM",
            "prior": {
                "type": "LinTransGaussian",
                "latent": {
                    "type": "Gaussian",
                    "mean": None,
                    "cov": "K",
                },
                "scale": 1.0,
                "shift": 0.0,
            },
            "postTrans": {
                "type": "LinSolveGaussian",
                "latent": {"type": "Prior"},
                "inv": "K",
                "explicit": True,
            },
        },
        "obs": {
            "type": "BFEMObservation",
            "models": ["solid", "load", "diri"],
            "init": {
                "type": "Init",
                "mesh": {"type": "manual", "file": "bar_coarse.mesh"},
            },
            "solver": {
                "type": "Linsolve",
            },
            "noise": None,
        },
        "bobs": {
            "type": "BoundaryObservation",
            "noise": None,
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

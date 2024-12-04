props = {
    "modules": ["init", "solve"],
    "init": {
        "type": "Init",
        "mesh": {"type": "gmsh", "file": "meshes/plate_r1.msh"},
        "nodeGroups": ["l", "lb"],
        "l": {"xtype": "min"},
        "lb": {"xtype": "min", "ytype": "min"},
    },
    "solve": {
        "type": "BFEMSolve",
        "fineSolve": {
            "type": "Linsolve",
            "tables": ["strain"],
        },
        "sequential": False,
    },
    "model": {
        "models": ["solid", "bfem", "load", "diri", "bobs", "obs"],
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
                "mesh": {"type": "gmsh", "file": "meshes/plate_r0.msh"},
            },
            "solver": {
                "type": "Linsolve",
                "tables": ["strain"],
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

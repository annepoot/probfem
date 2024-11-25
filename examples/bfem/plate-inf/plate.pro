modules = [ init, solver ];

init =
{
  type = Init;

  mesh =
  {
    type = gmsh;
    file = meshes/plate_r1.msh;
  };

  nodeGroups = [ l, lb ];

  l =
  {
    xtype = min;
  };

  lb =
  {
  	xtype = min;
    ytype = min;
  };
};

solver =
{
  type = BFEMInfSolve;

  nsample = 20;
  seed = 0;
};

model =
{
  models = [ solid, xsolid, load, diri, obs, ref ];

  solid =
  {
    type = Solid;

    elements = all;

    material =
    {
      type = Isotropic;
      rank = 2;
      anmodel = plane_stress;

      E = 3.;
      nu = 0.2;
    };

    shape =
    {
      type = Triangle3;
      intScheme = Gauss1;
    };
  };

  xsolid =
  {
    type = XSolid;

    elements = all;

    material =
    {
      type = Isotropic;
      rank = 2;
      anmodel = plane_stress;

      E = 3.;
      nu = 0.2;
    };

    shape =
    {
      type = Triangle3;
      intScheme = Gauss1;
    };
  };

  obs =
  {
    type = BFEMObservation;

    models = [ solid, load, diri ];

    init =
    {
      type = Init;

      mesh =
      {
        type = gmsh;
        file = meshes/plate_r0.msh;
      };
    };

    solver =
    {
      type = Linsolve;
    };

    noise = None;
  };

  ref =
  {
    type = BFEMReference;

    models = [ solid, load, diri ];

    init =
    {
      type = Init;

      mesh =
      {
        type = gmsh;
        file = meshes/plate_r1.msh;
      };
    };

    solver =
    {
      type = Linsolve;
      getMassMatrix = True;
    };
  };

  load =
  {
    type = Load;

    elements = all;

    dofs   = [ dx  ];
    values = [ 1.0 ];

    shape =
    {
      type = Triangle3;
      intScheme = Gauss1;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ l , lb ];
    dofs   = [ dx, dy ];
    values = [ 0., 0. ];
  };
};

modules = [ init, solver, conv ];

init =
{
  type = Init;

  mesh =
  {
    type = gmsh;
    file = meshes/box_r3.msh;
  };

  nodeGroups = [ l, r, b, t ];

  l =
  {
    xtype = min;
  };

  r =
  {
    xtype = max;
  };

  b =
  {
    ytype = min;
  };

  t =
  {
    ytype = max;
  };
};

solver =
{
  type = BFEMSolve;

  fineSolve = {
    type = Linsolve;

    tables = [ strain ];
  };

  sequential = 9;
};

conv =
{
  type = Conversion;

  sources = [ state0 ];
  targets = [ tables.state0 ];
  convTypes = [ field2table ];
};

model =
{
  models = [ solid, bfem, load, diri, bobs, obs ];

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
      type = Quad4;
      intScheme = Gauss9;
    };
  };

  bfem =
  {
    type = BFEM;

    prior =
    {
      type = LinTransGaussian;

      latent =
      {
        type = Gaussian;

        mean = None;
        cov = K;
      };

      scale = 1.0;
      shift = 0.0;
    };

    postTrans =
    {
      type = LinSolveGaussian;

      latent = {
        type = Prior;
      };

      inv = K;
      explicit = True;
    };
  };

  obs =
  {
    type = RandomBFEMObservation;

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
          file = meshes/box_r0.msh;
        };
      };

      solver =
      {
        type = Linsolve;
      };

      noise = None;
    };

    seed = 1;
    nobs = 10;
  };

  bobs =
  {
    type = BoundaryObservation;

    noise = None;
  };

  load =
  {
    type = Load;

    elements = all;

    dofs   = [ dx  ];
    values = [ 1.0 ];

    shape =
    {
      type = Quad4;
      intScheme = Gauss9;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ l , r , b , t , l , r , b , t  ];
    dofs   = [ dx, dx, dx, dx, dy, dy, dy, dy ];
    values = [ 0., 0., 0., 0., 0., 0., 0., 0. ];
  };
};

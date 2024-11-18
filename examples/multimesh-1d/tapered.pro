modules = [ init, solver, conv ];

init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = bar_fine.mesh;
  };

  nodeGroups = [ left, right ];

  left =
  {
    xtype = min;
  };

  right =
  {
    xtype = max;
  };
};

solver =
{
  type = BFEMSolve;

  fineSolve = {
    type = Linsolve;

    tables = [ strain ];
  };

  sequential = 5;
};

conv =
{
  type = Conversion;

  sources = [ state0 ];
  targets = [ tables.state0 ];
  convTypes = [ field2table ];
};

convobs =
{
  type = Conversion;

  sources = [ obs.obs.state0, obs.obs.tables.state0, tables.state0Coarse ];
  targets = [ obs.obs.tables.state0, tables.state0Coarse, tables.state0Error ];
  convTypes = [ field2table, coarse2fine, coarse2error ];
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
      type = Heterogeneous;
      rank = 1;
      anmodel = bar;

      E = 0.1 - 0.099 * x;
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss2;
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
          type = manual;
          file = bar_coarse.mesh;
        };
      };

      solver =
      {
        type = Linsolve;
      };

      noise = None;
    };

    nobs = 10;
    seed = 0;
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

    dofs   = [ dx ];
    values = [ 3. ];

    shape =
    {
      type = Line2;
      intScheme = Gauss2;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ left, right ];
    dofs   = [ dx, dx ];
    values = [ 0.0, 0.0 ];
  };
};

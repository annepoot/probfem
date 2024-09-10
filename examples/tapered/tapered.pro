init =
{
  type = BFEMInit;

  coarseInit =
  {
    type = Init;
    mesh =
    {
      type = manual;
      file = bar_coarse.mesh;
    };
  };

  mesh =
  {
    type = manual;
    file = bar_fine.mesh;
  };

  nodeGroups = [ left, right, mid ];

  left =
  {
    xtype = min;
  };

  right =
  {
    xtype = max;
  };

  mid =
  {
    xtype = mid;
  };
};

solve =
{
  type = BFEMSolve;

  coarseSolve={
    type = Linsolve;

    tables = [ strain ];
  };

  fineSolve = {
    type = Linsolve;

    tables = [ strain ];
  };

  sequential = True;
  nsample = 20;
};

conv =
{
  type = Conversion;

  sources = [ fine.state0, coarse.state0, coarse.tables.state0, fine.tables.state0Coarse ];
  targets = [ fine.tables.state0, coarse.tables.state0, fine.tables.state0Coarse, fine.tables.state0Error ];
  convTypes = [ field2table, field2table, coarse2fine, coarse2error ];
};

model =
{
  type = Multi;
  models = [ solid, bfem, bobs, obs, load, diri ];

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
        type = DirectGaussian;

        mean = 0;
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
    };
  };

  obs =
  {
    type = CGObservation;

    matrix = K;
    renormalize = True;
    nobs = None;
  };

  bobs =
  {
    type = BoundaryObservation;
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

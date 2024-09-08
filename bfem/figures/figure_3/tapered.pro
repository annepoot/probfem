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
};

model =
{
  type = Multi;
  models = [ solid, bfem, obs, bobs, load, diri ];

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
    type = BFEMObservation;
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

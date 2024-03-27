init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = 2nodebar.mesh;
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

rmfem =
{
  type = RMFem;

  solveModule =
  {
    type = Linsolve;
  };

  nsample = 50;
  seed = 0;
};

rmplot =
{
  type = RMPlot;

  figure =
  {
    title = Figure 2;
    xlabel = x;
    ylabel = Solution;
  };

  reference =
  {
    color = black;
    linewidth = 1;
  };

  perturbed =
  {
    color = grey;
    alpha = 0.3;
  };
};

model =
{
  type = Multi;
  models = [ solid, load, diri, rm ];

  solid =
  {
    type = Solid;

    elements = all;

    material =
    {
      type = Heterogeneous;
      rank = 1;
      anmodel = bar;

      E = 1;
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss2;
    };
  };

  load =
  {
    type = Load;

    elements = all;

    dofs   = [ dx ];

    // forcing term yields u(x) = sin(2*pi*x)
    // if kappa = 1
    values = [ 4*pi**2 * sin(2*pi*x) ];

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

  rm =
  {
    type = RandomMesh;

    p = 1;

    boundary =
    {
        groups = [ left, right ];
    };
  };
};

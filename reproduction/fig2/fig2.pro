init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = fig2.mesh;
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
    elemTables = [ strain, size ];
  };

  nsample = 50;
  seed = 0;

  errorTables = [ solution, strain ];
  estimatorTables = [ eta, eta1, eta2 ];
};

rmplot =
{
  type = RMPlot;
  plotType = node;

  field = solution;
  comp = dx;

  figure =
  {
    title = Figure 2;
    xlabel = x;
    ylabel = Solution;
  };

  exact =
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
  models = [ solid, load, diri, rm, ref ];

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
      intScheme = Gauss1;
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

  ref =
  {
    type = Reference;

    u = sin(2*pi*x);
    kappa = 1+0;
  };
};

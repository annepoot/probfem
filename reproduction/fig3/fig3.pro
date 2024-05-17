init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = fig3.mesh;
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

  nsample = 30;
  seed = 0;

  errorTables = [ solution, strain ];
  estimatorTables = [ eta1, eta2 ];
};

rmplot =
{
  type = RMPlot;
  plotType = node;

  field = solution;
  comp = dx;

  exact =
  {
    color = C0;
  };

  fem =
  {
    color = C1;
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

      E = 1 + x**3;
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

    // forcing term yields u(x) = (x**3) * sin(a*pi*x) * exp(-b*(x-0.5)**2)
    // if kappa = 1 + x**3
    values = [ - exp(-b*(x-0.5)**2) * ( (1+x**3)*x**2*(-a**2*pi**2*x*sin(a*pi*x) + (-4*b*x+b) * sin(a*pi*x) + a*pi*(3-b*x*(2*x-1))*cos(a*pi*x) + a*pi*cos(a*pi*x)) + (3*x**4 + 2 * (x**3+1) * x - 2 * b * (x-0.5)*(1+x**3) * x**2) * ((3 - b * x * (2*x-1))*sin(a*pi*x) + a*pi*x * cos(a*pi*x)) ) ];

    params =
    {
      a = 15;
      b = 50;
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss4;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ right ];
    dofs   = [ dx ];
    values = [ 0.0 ];
  };

  rm =
  {
    type = RandomMesh;

    p = 3;

    boundary =
    {
        groups = [ left, right ];
    };
  };

  ref =
  {
    type = Reference;

    u = x**3 * sin(a*pi*x) * exp(-b*(x-0.5)**2);
    kappa = 1 + x**3;

    params =
    {
      a = 15;
      b = 50;
    };
  };
};

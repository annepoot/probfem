init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = 2nodebar.mesh;
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

rmfem =
{
  type = RMFem;

  solveModule =
  {
    type = Linsolve;
  };

  nsample = 10;
  seed = 0;

  writeMesh =
  {
    type = manual;
    file = meshes/2nodebar-p{}.mesh;
  };
};

rmplot =
{
  type = RMPlot;

  figure =
  {
    title = Hello, this is a title;
  };

  reference =
  {
    color = C0;
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

    p = 3;

    boundary =
    {
        groups = [ left, right ];
    };
  };
};

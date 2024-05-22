init =
{
  type = Init;

  mesh =
  {
    type = manual;
    file = 1d-lin.mesh;
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
    type = MCMC;

    solveModule =
    {
      type = Linsolve;

      tables = [ stiffness ];
      elemTables = [ strain, size ];
    };

    variables = [solid.material.params.xi_1,
                 solid.material.params.xi_2,
                 solid.material.params.xi_3,
                 solid.material.params.xi_4];

    startValue = [0., 0., 0., 0.];

    priorStd = 1.;
    proposalStd = 1e-4;

    output = [variables, state0, tables.stiffness.];

    nsample = 30000;
    seed = 0;
  };

  nsample = 20;
  seed = 0;
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
  models = [ solid, load, diri, rm, ref, obs ];

  solid =
  {
    type = Solid;

    elements = all;

    material =
    {
      type = Heterogeneous;
      rank = 1;
      anmodel = bar;

      E = exp(sqrt(2) * (xi_1/(pi) * sin(pi*x) + xi_2/(2*pi) * sin(2*pi*x) + xi_3/(3*pi) * sin(3*pi*x) + xi_4/(4*pi) * sin(4*pi*x)));

      params = {
        xi_1 = 0.00;
        xi_2 = 0.00;
        xi_3 = 0.00;
        xi_4 = 0.00;
      };
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss4;
    };
  };

  load =
  {
    type = Load;
    elements = all;

    dofs   = [ dx ];
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

  obs =
  {
    type = Observation;
    field = state0;

    observation = {
      type = equalLocation;
      nobs = 4;
      includeBoundary = False;
    };

    measurement = {
      type = direct;
      values = [0.9510565162951536, 0.5877852522924731, -0.5877852522924729, -0.9510565162951536];

      corruption = {
        type = multivariate_normal;
        cov = 1e-100;
        seed = 0;
      };
    };

    noise = {
      type = multivariate_normal;
      cov = 1e-20;
    };
  };

  ref =
  {
    type = Reference;

    u = sin(2*pi*x);
    kappa = 1+0;
  };
};

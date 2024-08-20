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

mcmc =
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

  startValue = [1., 1., 0.25, 0.25];

  priorStd = 1.;
  proposalStd = 1e-4;

  output = [variables, state0, tables.stiffness.];

  nsample = 10000;
  seed = 0;
};

model =
{
  type = Multi;
  models = [ solid, load, diri, obs ];

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
        xi_1 = 1.0;
        xi_2 = 1.0;
        xi_3 = 0.25;
        xi_4 = 0.25;
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
    values = [ sin(2*pi*x) ];

    shape =
    {
      type = Line2;
      intScheme = Gauss4;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ left, right ];
    dofs   = [ dx, dx ];
    values = [ 0.0, 0.0 ];
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
//       values = [0.01154101, 0.01667733, 0.01592942, 0.00980423, -0.00043005, -0.01177105, -0.02001336, -0.0211289, -0.01350695];
      values = [0.01667733, 0.00980423, -0.01177105, -0.0211289];

      corruption = {
        type = multivariate_normal;
        cov = 1e-8;
        seed = 0;
      };
    };

    noise = {
      type = multivariate_normal;
      cov = 1e-8;
    };
  };
};

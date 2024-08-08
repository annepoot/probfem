init =
{
  type = Init;

  mesh =
  {
    type = gmsh;
    file = meshes/bar_r0.msh;
  };

  nodeGroups = [ left, right, top, bottom ];

  left =
  {
    xtype = min;
  };

  right =
  {
    xtype = max;
  };

  top =
  {
    ytype = max;
  };

  bottom =
  {
    ytype = min;
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
      elemTables = [ size ];
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

  nsample = 20;
  seed = 0;

//  errorTables = [ solution, strain ];
//  estimatorTables = [ eta1, eta2 ];
};

model =
{
  type = Multi;
  models = [ solid, load, diri, rm, obs ];

  solid =
  {
    type = Solid;

    elements = all;

    material =
    {
      type = Heterogeneous;
      rank = 2;
      anmodel = plane_strain;

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
      type = Quad4;
      intScheme = Gauss9;
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
      type = Quad4;
      intScheme = Gauss9;
    };
  };

  diri =
  {
    type = Dirichlet;

    groups = [ left, right, left, right ];
    dofs   = [ dx, dx, dy, dy ];
    values = [ 0.0, 0.0, 0.0, 0.0 ];
  };

  rm =
  {
    type = RandomMesh;

    p = 1;

    boundary =
    {
      groups = [ left, right, top, bottom ];
      dofs = [ dx, dx, dy, dy ];
    };
  };

  obs =
  {
    type = Observation;
    field = state0;

    observation = {
      type = directLocation;
      locs = [[0.1, 0.05],
              [0.2, 0.05],
              [0.3, 0.05],
              [0.4, 0.05],
              [0.5, 0.05],
              [0.6, 0.05],
              [0.7, 0.05],
              [0.8, 0.05],
              [0.9, 0.05]];
      dofs = dx;
    };

    measurement = {
      type = direct;
      values = [ 0.01154103, 0.01667735, 0.01592944, 0.00980424, -0.00043006, -0.01177108, -0.02001339, -0.02112894, -0.01350698];

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

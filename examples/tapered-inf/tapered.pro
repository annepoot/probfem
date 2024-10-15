modules = [ init, solver ];

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
  type = BFEMInfSolve;

  nsample = 20;
  seed = 0;
};

model =
{
  models = [ solid, xsolid, load, diri, obs, ref ];

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
      rho = 1.0;
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss2;
    };
  };

  xsolid =
  {
    type = XSolid;

    elements = all;

    material =
    {
      type = Heterogeneous;
      rank = 1;
      anmodel = bar;

      E = 0.1 - 0.099 * x;
      rho = 1.0;
    };

    shape =
    {
      type = Line2;
      intScheme = Gauss2;
    };
  };

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

  ref =
  {
    type = BFEMReference;

    models = [ solid, load, diri ];

    init =
    {
      type = Init;

      mesh =
      {
        type = manual;
        file = bar_fine.mesh;
      };
    };

    solver =
    {
      type = Linsolve;
      getMassMatrix = True;
    };
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

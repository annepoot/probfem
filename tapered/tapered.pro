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

  title = "Plot";

  figure =
  {

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

      E = 1.0 - 0.99 * x;
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
    values = [ 0.0, 1.0 ];
  };

  rm =
  {
    type = RandomMesh;
  };
};

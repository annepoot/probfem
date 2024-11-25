init =
{
  type = Init;

  mesh =
  {
    type = gmsh;
    file = meshes/bar_r0.msh;
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
  type = Linsolve;
  tables = [ stiffness ];
};

model =
{
  type = Multi;
  models = [ solid, load, diri ];

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
};

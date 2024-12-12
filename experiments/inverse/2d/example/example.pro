log =
{
  pattern = "*.info";
  file    = "-$(CASE_NAME).log";
};

control =
{
  runWhile = "i<1";
  fgMode = false;
};

userinput =
{
  modules = [ "input", "ngroups" ];

  input =
  {
    type = "Input";
    file = "simple.mesh";
  };

  ngroups =
  {
    type = "GroupInput";
    nodeGroups = [ "left", "right", "top", "bot" ];

    left.xtype = "min";
    right.xtype = "max";
    top.ytype = "max";
    bot.ytype = "min";
  };
};

model =
{
  type   = "Matrix";

  model =
  {
    type   = "Multi";
    models = [ "elastic", "diri" ];

    diri =
    {
      type = "Dirichlet";

      initDisp = 0.1;
      dispIncr = 0.0;

      nodeGroups = [ "left", "bot", "right", "top" ];
      dofs = [ "dx", "dy", "dx", "dy" ];
      loaded =  2; 
    };

    elastic =
    {
      type     = "Elastic";
      elements = "all";

      material =
      {
        type = "LinearIsotropic";

        anmodel = "PLANE_STRAIN";
        rank = 2;

        E = 1000.;
        nu = 0.2;
      };

      shape =
      {
        type      = "Quad4";
        intScheme = "Gauss2*Gauss2";
      };
    };
  };
};

usermodules =
{
  modules = [ "solver" ];

  solver =
  {
    type = "Linsolve";
  };
};

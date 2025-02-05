from .fem_props import get_fem_props

__all__ = ["get_fem_2d_props"]


def get_fem_2d_props():
    fem_props = get_fem_props()

    # fem_props["log"]["rank"] = 0

    ngroups_props = fem_props["userinput"]["ngroups"]
    ngroups_props["nodeGroups"].append("bottomleft")
    ngroups_props["bottomleft.xtype"] = "min"
    ngroups_props["bottomleft.ytype"] = "min"

    elastic_props = fem_props["model"]["model"]["elastic"]
    elastic_props["shape"]["type"] = "Quad4"
    elastic_props["shape"]["intScheme"] = "Gauss4*Gauss4"

    material_props = elastic_props["material"]
    material_props["rank"] = 2
    material_props["anmodel"] = "PLANE_STRESS"
    material_props["nu"] = 0.0
    material_props.pop("area")

    diri_props = fem_props["model"]["model"]["diri"]
    diri_props["nodeGroups"].append("bottomleft")
    diri_props["dofs"].append("dy")

    return fem_props

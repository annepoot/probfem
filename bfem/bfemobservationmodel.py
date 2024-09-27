import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import split_off_type


class BFEMObservationModel(Model):
    def RETURNMATRICES(self, globdat):
        self._init.run(self._obsdat)
        self._solver.run(self._obsdat)
        globdat["obs"][self.get_name()] = self._obsdat

    def GETOBSERVATIONS(self, globdat):
        Phi = self._get_phi(globdat)
        Phic = self._constrain_phi(Phi, globdat)
        return Phic, Phic.T @ globdat["extForce"], self._noise

    @Model.save_config
    def configure(self, globdat, init, models, solver, noise):

        inittype, newinitprops = split_off_type(init)
        solvertype, newsolverprops = split_off_type(solver)
        self._init = globdat[gn.MODULEFACTORY].get_module(inittype, "obsinit")
        self._solver = globdat[gn.MODULEFACTORY].get_module(solvertype, "obssolve")

        initprops = {}
        solverprops = {}
        for module in globdat[gn.MODULES].values():
            if isinstance(module, type(self._init)):
                if len(initprops) > 0:
                    raise ValueError("ambiguous init props")
                initprops.update(module.get_config())
            elif isinstance(module, type(self._solver)):
                if len(solverprops) > 0:
                    raise ValueError("ambiguous solver props")
                solverprops.update(module.get_config())

        initprops.update(newinitprops)
        solverprops.update(newsolverprops)

        self._obsdat = {
            gn.MODULEFACTORY: globdat[gn.MODULEFACTORY],
            gn.MODELFACTORY: globdat[gn.MODELFACTORY],
            gn.SHAPEFACTORY: globdat[gn.SHAPEFACTORY],
            gn.SOLVERFACTORY: globdat[gn.SOLVERFACTORY],
            gn.PRECONFACTORY: globdat[gn.PRECONFACTORY],
        }

        self._init.configure(self._obsdat, **initprops)
        self._solver.configure(self._obsdat, **solverprops)

        model_list = []
        modelprops = {}
        modelprops["models"] = models
        for model_name in models:
            for m in globdat[gn.MODELS].values():
                if m.get_name() == model_name:
                    model_list.append(m)
                    modelprops[model_name] = m.get_config()
                    break
            else:
                raise ValueError("Model '{}' not found!".format(model_name))

        self._init.init(self._obsdat, modelprops=modelprops)
        self._solver.init(self._obsdat)

        self._noise = noise

    def _get_phi(self, globdat):
        cglobdat = self._obsdat
        fglobdat = globdat

        elemsc = cglobdat[gn.ESET]
        nodesc = cglobdat[gn.NSET]
        nodes = fglobdat[gn.NSET]
        dofsc = cglobdat[gn.DOFSPACE]
        dofs = fglobdat[gn.DOFSPACE]

        rank = fglobdat[gn.MESHRANK]
        shapefac = cglobdat[gn.SHAPEFACTORY]
        shape = shapefac.get_shape(cglobdat[gn.MESHSHAPE], "Gauss1")
        dof_types = fglobdat[gn.DOFSPACE].get_types()

        Phi = np.zeros((dofs.dof_count(), dofsc.dof_count()))

        # Go over the coarse mesh
        for inodesc in elemsc:
            coordsc = nodesc.get_some_coords(inodesc)

            # Get the bounding box of the coarse element
            bbox = np.zeros((rank, 2))
            for i in range(rank):
                bbox[i, 0] = min(coordsc[:, i])
                bbox[i, 1] = max(coordsc[:, i])

            # Go over the fine mesh
            for inode, coords in enumerate(nodes):

                # Check if the node falls inside the bounding box
                inside = True
                for i in range(rank):
                    if coords[i] < bbox[i, 0] or coords[i] > bbox[i, 1]:
                        inside = False
                        break

                # If so, check if the node falls inside the shape itself
                if inside:
                    loc_point = shape.get_local_point(coords, coordsc)
                    inside = shape.contains_local_point(loc_point, tol=1e-8)

                # If so, add the relative shape function values to the Phi matrix
                if inside:
                    svals = np.round(shape.eval_shape_functions(loc_point), 12)
                    idofs = dofs.get_dofs([inode], dof_types)

                    for i, inodec in enumerate(inodesc):
                        sval = svals[i]
                        idofsc = dofsc.get_dofs([inodec], dof_types)
                        Phi[idofs, idofsc] = sval

        return Phi

    def _constrain_phi(self, Phi, globdat):
        Phic = Phi.copy()
        cdofs, _ = self._obsdat[gn.CONSTRAINTS].get_constraints()

        for cdof in cdofs:
            mask = np.isclose(Phi[:, cdof], 1.0)
            if sum(mask) > 0:
                Phic[mask] = 0.0
                Phic[:, cdof] = 0.0

        return Phic

import numpy as np

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import split_off_type
from myjive.fem import to_xnodeset


class RandomBFEMObservationModel(Model):
    def RETURNMATRICES(self, globdat):
        for i, obsmodel in enumerate(self._obslist):
            if globdat[gn.MESHRANK] == 1:
                self._obslist[i] = self._randomize_model_1d(obsmodel, globdat)
            elif globdat[gn.MESHRANK] == 2:
                self._obslist[i] = self._randomize_model_2d(obsmodel, globdat)
            else:
                raise ValueError("observation model is only implemented for 1d and 2d")

        for obsmodel in self._obslist:
            obsmodel.RETURNMATRICES(globdat)

    def GETOBSERVATIONS(self, globdat):
        Phi, measurement, noise = self._get_phi(globdat)
        assert np.allclose(Phi.T @ globdat["extForce"], measurement)
        return Phi, measurement, noise

    @Model.save_config
    def configure(self, globdat, obs, nobs, seed):
        self._nobs = nobs
        self._rng = np.random.default_rng(seed)

        self._obslist = []

        obstype, obsprops = split_off_type(obs)

        for i in range(self._nobs):
            obsmodel = globdat[gn.MODELFACTORY].get_model(obstype, "obs{}".format(i))
            obsmodel.configure(globdat, **obsprops)
            self._obslist.append(obsmodel)

    def _get_phi(self, globdat):
        Phis = []
        measurements = []

        for i, obsmodel in enumerate(self._obslist):
            Phi, measurement, noise = obsmodel.GETOBSERVATIONS(globdat)
            Phis.append(Phi)
            measurements.append(measurement)
            if i == 0:
                groupnoise = noise
            else:
                if noise != groupnoise:
                    raise ValueError("Incompatible noises")

        Phi = np.concatenate(Phis, axis=1)
        measurement = np.concatenate(measurements)

        return Phi, measurement, groupnoise

    def _randomize_model_1d(self, obsmodel, globdat):
        cnodes = obsmodel._obsdat[gn.NSET]
        fnodes = globdat[gn.NSET]

        selection = self._rng.choice(
            np.arange(1, len(fnodes) - 1), size=len(cnodes) - 2, replace=False
        )
        selection = np.concatenate(
            (np.array([0]), np.sort(selection), np.array([len(fnodes) - 1]))
        )

        new_coords = fnodes.get_some_coords(selection)

        xnodes = to_xnodeset(cnodes)
        xnodes.set_coords(new_coords)
        cnodes = xnodes.to_nodeset()

        assert obsmodel._obsdat[gn.NSET] == cnodes

        return obsmodel

    def _randomize_model_2d(self, obsmodel, globdat):
        nodes = obsmodel._obsdat[gn.NSET]

        # new_coords = self._rng.random(2)

        # xnodes = to_xnodeset(nodes)
        # for i, coords in enumerate(nodes):
        #     if not np.isclose(coords[0], 0.0) and not np.isclose(coords[0], 1.0):
        #         coords[0] = new_coords[0]
        #     if not np.isclose(coords[1], 0.0) and not np.isclose(coords[1], 1.0):
        #         coords[1] = new_coords[1]
        #     xnodes.set_node_coords(i, coords)
        # nodes = xnodes.to_nodeset()

        xnodes = to_xnodeset(nodes)
        for i, coords in enumerate(nodes):
            if np.isclose(coords[0], 0.0):
                pass
            elif np.isclose(coords[1], 0.0):
                pass
            elif np.isclose(coords[0], 1.0):
                pass
            elif np.isclose(coords[1], 1.0):
                pass
            else:
                coords = 0.25 + 0.5 * self._rng.random(2)
                # coords = self._rng.random(2)
                xnodes.set_node_coords(i, coords)
        nodes = xnodes.to_nodeset()

        assert obsmodel._obsdat[gn.NSET] == nodes

        return obsmodel

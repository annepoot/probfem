import numpy as np
import os
from scipy.stats import multivariate_normal

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import check_dict, check_list, split_off_type, get_recursive


class ObservationModel(Model):
    def GETPREDICTION(self, globdat):
        prediction = self._get_prediction(globdat)
        return prediction

    def GETMEASUREMENTS(self, globdat):
        measurements = self._get_measurements(globdat)
        return measurements

    def GETLIKELIHOOD(self, globdat):
        likelihood = self._get_likelihood(globdat)
        return likelihood

    def GETLOGLIKELIHOOD(self, globdat):
        loglikelihood = self._get_loglikelihood(globdat)
        return loglikelihood

    def ADDLOGLIKELIHOOD(self, loglikelihood, globdat):
        loglikelihood += self._get_loglikelihood(globdat)
        return loglikelihood

    def MULTIPLYLIKELIHOOD(self, likelihood, globdat):
        likelihood *= self._get_likelihood(globdat)
        return likelihood

    @Model.save_config
    def configure(self, globdat, *, field, observation, measurement, noise):
        # Validate input arguments
        check_dict(self, observation, keys=["type"])
        check_dict(self, measurement, keys=["type"])
        check_dict(self, noise, keys=["type", "cov"])
        self._fieldname = field

        typ, observationprops = split_off_type(observation)
        self._observationfunc = self._get_observation_func(typ)
        self._observationprops = observationprops

        typ, measurementprops = split_off_type(measurement)
        measurementfunc = self._get_measurement_func(typ)
        measurementprops = measurementprops
        self._measurements = measurementfunc(**measurementprops)

        typ, noiseprops = split_off_type(noise)
        self._noisefunc = self._get_distribution(typ)
        self._noiseprops = noiseprops

    def _get_prediction(self, globdat):
        if "." in self._fieldname:
            full_field = get_recursive(globdat, self._fieldname.split("."))
        else:
            full_field = globdat[self._fieldname]
        prediction = self._observationfunc(
            globdat, full_field, **self._observationprops
        )
        return prediction

    def _get_measurements(self, globdat):
        return self._measurements

    def _get_loglikelihood(self, globdat):
        prediction = self._get_prediction(globdat)
        measurements = self._get_measurements(globdat)
        loglikelihood = self._noisefunc(prediction, **self._noiseprops).logpdf(
            measurements
        )
        return loglikelihood

    def _get_likelihood(self, globdat):
        loglikelihood = self._get_loglikelihood(globdat)
        return np.exp(loglikelihood)

    def _get_observation_func(self, typ):
        if typ == "directSelection":
            return self.direct_selection
        elif typ == "equalSelection":
            return self.equal_selection
        elif typ == "directLocation":
            return self.direct_location
        elif typ == "equalLocation":
            return self.equal_location
        else:
            raise ValueError("'{}' is not a valid observation function".format(typ))

    def _get_measurement_func(self, typ):
        if typ == "direct":
            return self.direct_measurement
        elif typ == "generative":
            return self.generative_measurements
        else:
            raise ValueError("'{}' is not a valid measurement function".format(typ))

    def _get_distribution(self, typ):
        if typ == "multivariate_normal":
            return multivariate_normal
        else:
            raise ValueError("'{}' is not a valid noise function".format(typ))

    def direct_selection(self, globdat, full_field, *, dofs):
        return full_field[dofs]

    def equal_selection(self, globdat, full_field, *, nobs, includeBoundary):
        ndofs = len(full_field)
        if includeBoundary:
            if (ndofs - 1) % (nobs - 1) != 0:
                raise ValueError("nobs incompatible with mesh")
            dofs = np.arange(0, ndofs)[:: (ndofs - 1) // (nobs - 1)]
        else:
            if (ndofs - 1) % (nobs + 1) != 0:
                raise ValueError("nobs incompatible with mesh")
            dofs = np.arange(0, ndofs)[:: (ndofs - 1) // (nobs + 1)][1:-1]
        return self.direct_selection(globdat, full_field, dofs=dofs)

    def direct_location(self, globdat, full_field, *, locs, dofs):
        elems = globdat[gn.ESET]
        nodes = globdat[gn.NSET]
        dofspace = globdat[gn.DOFSPACE]
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss1")

        nobs = len(locs)
        pred_field = np.zeros(nobs)
        if isinstance(dofs, str):
            dofs = [dofs] * nobs

        for i, (loc, dof) in enumerate(zip(locs, dofs)):
            for inodes in elems:
                coords = nodes.get_some_coords(inodes)

                if shape.contains_global_point(loc, coords):
                    idofs = dofspace.get_dofs(inodes, [dof])
                    elfield = full_field[idofs]
                    sfuncs = shape.eval_global_shape_functions(loc, coords)
                    pred_field[i] = sfuncs @ elfield
                    break
            else:
                raise RuntimeError("No matching element found!")

        return pred_field

    def equal_location(self, globdat, full_field, *, nobs, dofs, includeBoundary):
        if includeBoundary:
            locs = np.linspace(0, 1, nobs)
        else:
            locs = np.linspace(0, 1, nobs + 2)[1:-1]
        locs = locs.reshape((-1, 1))
        return self.direct_location(globdat, full_field, locs=locs, dofs=dofs)

    def direct_measurement(self, *, values, corruption={}):
        if len(corruption) > 0:
            typ, corruptionprops = split_off_type(corruption)
            distribution = self._get_distribution(typ)(**corruptionprops)
            return values + distribution.rvs(size=len(values))
        else:
            return values

    def generative_measurements(self):
        raise NotImplementedError("")

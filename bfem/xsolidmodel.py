import numpy as np
from numba import njit

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjivex.materials import new_material
import myjive.util.proputils as pu
from myjive.util.proputils import check_dict, split_off_type
from .hypermesh import create_hypermesh

TYPE = "type"
INTSCHEME = "intScheme"
DOFTYPES = ["dx", "dy", "dz"]

__all__ = ["XSolidModel"]


class XSolidModel(Model):
    def GETXMATRIX0(self, K, globdat1, globdat2, **kwargs):
        K = self._get_cross_stiff_matrix(K, globdat1, globdat2, **kwargs)
        return K

    def GETXMATRIX2(self, M, globdat1, globdat2, **kwargs):
        M = self._get_cross_mass_matrix(M, globdat1, globdat2, **kwargs)
        return M

    @Model.save_config
    def configure(self, globdat, *, shape, elements, material, thickness=1.0):
        # Validate input arguments
        check_dict(self, shape, [TYPE, INTSCHEME])
        check_dict(self, material, [TYPE])
        self._thickness = thickness

        # Configure the material
        mattype, matprops = split_off_type(material)
        self._mat = new_material(mattype, "material")
        self._mat.configure(globdat, **matprops)
        self._config["material"] = self._mat.get_config()

        # Get shape and element info
        self._shape = globdat[gn.SHAPEFACTORY].get_shape(shape[TYPE], shape[INTSCHEME])
        self._egroup = elements

        # Make sure the shape rank and mesh rank are identitcal
        if self._shape.global_rank() != globdat[gn.MESHRANK]:
            raise RuntimeError("ElasticModel: Shape rank must agree with mesh rank")

        # Get basic dimensionality info
        self._rank = self._shape.global_rank()
        self._ipcount = self._shape.ipoint_count()
        self._dofcount = self._rank * self._shape.node_count()
        self._strcount = self._rank * (self._rank + 1) // 2  # 1-->1, 2-->3, 3-->6

        if self._rank == 2:
            self._thickness = pu.soft_cast(self._thickness, float)

    def _get_cross_stiff_matrix(self, K, globdat1, globdat2):
        egroup1 = globdat1[gn.EGROUPS][self._egroup]
        egroup2 = globdat2[gn.EGROUPS][self._egroup]

        elems1 = egroup1.get_elements()
        elems2 = egroup2.get_elements()

        elemsh, elemmap = create_hypermesh(elems1, elems2)

        nodes1 = elems1.get_nodes()
        nodes2 = elems2.get_nodes()
        nodesh = elemsh.get_nodes()

        for ielemh, inodesh in enumerate(elemsh):
            # Get the nodal coordinates of each element
            ielem1, ielem2 = elemmap[ielemh]

            inodes1 = elems1.get_elem_nodes(ielem1)
            inodes2 = elems2.get_elem_nodes(ielem2)
            inodesh = elemsh.get_elem_nodes(ielemh)

            idofs1 = globdat1[gn.DOFSPACE].get_dofs(inodes1, DOFTYPES[0 : self._rank])
            idofs2 = globdat2[gn.DOFSPACE].get_dofs(inodes2, DOFTYPES[0 : self._rank])

            coords1 = nodes1.get_some_coords(inodes1)
            coords2 = nodes2.get_some_coords(inodes2)
            coordsh = nodesh.get_some_coords(inodesh)

            # Get the gradients, weights and coordinates of each integration point
            ipoints = self._shape.get_global_integration_points(coordsh)
            weights = self._shape.get_integration_weights(coordsh)

            if self._rank == 2:
                weights *= self._thickness

            # Reset the element stiffness matrix
            elmat = np.zeros((self._dofcount, self._dofcount))

            for ipoint, weight in zip(ipoints, weights):
                # Get the B and D matrices for each integration point
                grads1 = self._shape.eval_global_shape_gradients(ipoint, coords1)
                grads2 = self._shape.eval_global_shape_gradients(ipoint, coords2)

                B_elem1 = self._get_B_matrix(grads1)
                B_elem2 = self._get_B_matrix(grads2)
                D_elem = self._mat.stiff_at_point(ipoint)

                # Compute the element stiffness matrix
                elmat += weight * B_elem1.T @ D_elem @ B_elem2

            # Add the element stiffness matrix to the global stiffness matrix
            K[np.ix_(idofs1, idofs2)] += elmat

        return K

    def _get_cross_mass_matrix(self, M, globdat1, globdat2):
        egroup1 = globdat1[gn.EGROUPS][self._egroup]
        egroup2 = globdat2[gn.EGROUPS][self._egroup]

        elems1 = egroup1.get_elements()
        elems2 = egroup2.get_elements()

        elemsh, elemmap = self._get_hypermesh(elems1, elems2)

        nodes1 = elems1.get_nodes()
        nodes2 = elems2.get_nodes()
        nodesh = elemsh.get_nodes()

        for ielemh, inodesh in enumerate(elemsh):
            # Get the nodal coordinates of each element
            ielem1, ielem2 = elemmap[ielemh]

            inodes1 = elems1.get_elem_nodes(ielem1)
            inodes2 = elems2.get_elem_nodes(ielem2)
            inodesh = elemsh.get_elem_nodes(ielemh)

            idofs1 = globdat1[gn.DOFSPACE].get_dofs(inodes1, DOFTYPES[0 : self._rank])
            idofs2 = globdat2[gn.DOFSPACE].get_dofs(inodes2, DOFTYPES[0 : self._rank])

            coords1 = nodes1.get_some_coords(inodes1)
            coords2 = nodes2.get_some_coords(inodes2)
            coordsh = nodesh.get_some_coords(inodesh)

            # Get the gradients, weights and coordinates of each integration point
            ipoints = self._shape.get_global_integration_points(coordsh)
            weights = self._shape.get_integration_weights(coordsh)

            if self._rank == 2:
                weights *= self._thickness

            # Reset the element stiffness matrix
            elmat = np.zeros((self._dofcount, self._dofcount))

            for ipoint, weight in zip(ipoints, weights):
                # Get the B and D matrices for each integration point
                sfuncs1 = self._shape.eval_global_shape_functions(ipoint, coords1)
                sfuncs2 = self._shape.eval_global_shape_functions(ipoint, coords2)

                N_elem1 = self._get_N_matrix(sfuncs1)
                N_elem2 = self._get_N_matrix(sfuncs2)
                M_elem = self._mat.mass_at_point(ipoint)

                # Compute the element stiffness matrix
                elmat += weight * N_elem1.T @ M_elem @ N_elem2

            # Add the element stiffness matrix to the global stiffness matrix
            M[np.ix_(idofs1, idofs2)] += elmat

        return M

    def _get_N_matrix(self, sfuncs):
        return self._get_N_matrix_jit(sfuncs, self._dofcount, self._rank)

    @staticmethod
    @njit
    def _get_N_matrix_jit(sfuncs, _dofcount, _rank):
        N = np.zeros((_rank, _dofcount))
        for i in range(_rank):
            N[i, i::_rank] = sfuncs
        return N

    def _get_B_matrix(self, grads):
        return self._get_B_matrix_jit(
            grads, self._strcount, self._dofcount, self._shape.node_count(), self._rank
        )

    @staticmethod
    @njit
    def _get_B_matrix_jit(grads, _strcount, _dofcount, _nodecount, _rank):
        B = np.zeros((_strcount, _dofcount))
        if _rank == 1:
            B = grads
        elif _rank == 2:
            for inode in range(_nodecount):
                i = 2 * inode
                gi = grads[:, inode]
                B[0, i + 0] = gi[0]
                B[1, i + 1] = gi[1]
                B[2, i + 0] = gi[1]
                B[2, i + 1] = gi[0]
        elif _rank == 3:
            for inode in range(_nodecount):
                i = 3 * inode
                gi = grads[:, inode]
                B[0, i + 0] = gi[0]
                B[1, i + 1] = gi[1]
                B[2, i + 2] = gi[2]
                B[3, i + 0] = gi[1]
                B[3, i + 1] = gi[0]
                B[4, i + 1] = gi[2]
                B[4, i + 2] = gi[1]
                B[5, i + 0] = gi[2]
                B[5, i + 2] = gi[0]
        return B

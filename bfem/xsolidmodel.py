import numpy as np
from numba import njit

from myjive.fem import XNodeSet, XElementSet
from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjivex.materials import new_material
import myjive.util.proputils as pu
from myjive.util.proputils import check_dict, split_off_type

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

    def _get_hypermesh(self, elems1, elems2):
        nodes1 = elems1.get_nodes()
        nodes2 = elems2.get_nodes()

        nodesh = XNodeSet()
        nodemap1 = np.zeros(len(nodes1), dtype=int)
        nodemap2 = np.zeros(len(nodes2), dtype=int)

        coords1 = nodes1.get_coords()
        coords2 = nodes2.get_coords()

        rank = coords1.shape[1]
        if rank != coords2.shape[1]:
            raise RuntimeError("incompatible rank!")

        for inode1, coords in enumerate(nodes1):
            inodeh = nodesh.add_node(coords)
            nodemap1[inode1] = inodeh

        for inode2, coords in enumerate(nodes2):
            mask = np.all(np.isclose(coords, coords1), axis=1)
            if np.sum(mask) == 0:
                inodeh = nodesh.add_node(coords)
            else:
                inode1 = np.where(mask)[0]
                if len(inode1) != 1:
                    raise RuntimeError("no unique matching node found")
                inodeh = nodemap1[inode1[0]]
            nodemap2[inode2] = inodeh

        elemsh = XElementSet(nodesh)
        elemmap = []

        for ielem1, inodes1 in enumerate(elems1):
            coords1 = nodes1.get_some_coords(inodes1)
            bbox1 = np.stack([np.min(coords1, axis=0), np.max(coords1, axis=0)])
            for ielem2, inodes2 in enumerate(elems2):
                coords2 = nodes2.get_some_coords(inodes2)
                bbox2 = np.stack([np.min(coords2, axis=0), np.max(coords2, axis=0)])

                # check bounding boxes
                if np.any(bbox1[0] > bbox2[1]) or np.any(bbox2[0] > bbox1[1]):
                    continue

                # check overlap
                if rank == 1:
                    if coords1[0, 0] > coords2[0, 0]:
                        left = coords1[0, 0]
                        ileft = nodemap1[inodes1[0]]
                    else:
                        left = coords2[0, 0]
                        ileft = nodemap2[inodes2[0]]

                    if coords1[1, 0] < coords2[1, 0]:
                        right = coords1[1, 0]
                        iright = nodemap1[inodes1[1]]
                    else:
                        right = coords2[1, 0]
                        iright = nodemap2[inodes2[1]]

                    if left < right:
                        elemsh.add_element([ileft, iright])
                        elemmap.append((ielem1, ielem2))

                elif rank == 2:
                    intersection = self._clip_polygons(coords1, coords2)
                    nside = len(intersection)

                    if nside == 0:
                        continue

                    elif nside >= 3:
                        # add one or more elements
                        # triangulation is done as:
                        # (0, 1, 2)
                        # (0, 2, 3)
                        # ...
                        # (0, n-2, n-1)
                        for isubelem in range(nside - 2):
                            indices = [0, isubelem + 1, isubelem + 2]
                            coordsh = nodesh.get_coords()
                            inodesh = np.zeros(3, dtype=int)

                            for i, coord in enumerate(intersection[indices]):
                                mask = np.all(np.isclose(coordsh, coord), axis=1)
                                if np.sum(mask) == 0:
                                    inodeh = nodesh.add_node(coord)
                                else:
                                    inodeh = np.where(mask)[0]
                                    if len(inodeh) != 1:
                                        raise RuntimeError(
                                            "no unique matching node found"
                                        )
                                    inodeh = inodeh[0]
                                inodesh[i] = inodeh

                            elemsh.add_element(inodesh)
                            elemmap.append((ielem1, ielem2))

                    else:
                        raise RuntimeError("degenerate polygon")

                else:
                    raise NotImplementedError("rank {} is not implemented".format(rank))

        if len(elemsh) != len(elemmap):
            raise RuntimeError("elemmap size mismatch")

        return elemsh, elemmap

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

    @staticmethod
    def _clip_polygons(coords1, coords2):
        clip = coords1.copy()

        for A, B in zip(coords2, np.roll(coords2, shift=-1, axis=0)):
            # Check each point in the clipped polygon
            cross = np.cross(B - A, clip - A)
            keep = np.logical_or(cross > 0, np.isclose(cross, 0))

            # If not all nodes should be kept, perform
            if np.all(np.logical_not(keep)):
                return np.zeros((0, 2))
            elif not np.all(keep):
                newclip = []
                for keepprev, keepcurr, coordsprev, coordscurr in zip(
                    np.roll(keep, shift=1, axis=0),
                    keep,
                    np.roll(clip, shift=1, axis=0),
                    clip,
                ):
                    if keepcurr:
                        if not keepprev:
                            mat = np.column_stack([B - A, -(coordscurr - coordsprev)])
                            vec = coordsprev - A
                            s, t = np.linalg.solve(mat, vec)
                            xcoords = A + s * (B - A)
                            assert np.allclose(
                                (1 - s) * A + s * B,
                                (1 - t) * coordsprev + t * coordscurr,
                            )
                            newclip.append(xcoords)
                        newclip.append(coordscurr)
                    else:
                        if keepprev:
                            # search for the intersection
                            mat = np.column_stack([B - A, -(coordscurr - coordsprev)])
                            vec = coordsprev - A
                            s, t = np.linalg.solve(mat, vec)
                            xcoords = A + s * (B - A)
                            assert np.allclose(
                                (1 - s) * A + s * B,
                                (1 - t) * coordsprev + t * coordscurr,
                            )
                            newclip.append(xcoords)

                clip = np.array(newclip)
                round_clip = np.round(clip, decimals=8)
                unique_idx = np.unique(round_clip, axis=0, return_index=True)[1]
                clip = clip[np.sort(unique_idx)]

                if len(clip) < 3:
                    return np.zeros((0, 2))

        return clip

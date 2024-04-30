import numpy as np
import os
from scipy.integrate import quad

from myjive.names import GlobNames as gn
from myjive.model.model import Model
from myjive.util.proputils import check_dict, check_list
from myjive.util import to_xtable


class RandomMeshModel(Model):
    def PERTURBNODES(
        self, nodes, globdat, meshsize, rng=np.random.default_rng(), **kwargs
    ):
        nodes = self._perturb_nodes(nodes, globdat, meshsize, rng=rng)
        return nodes

    def COMPUTEESTIMATOR(self, name, table, globdat, **kwargs):
        if name == "eta1":
            table = self._compute_estimator_1(table, globdat)
        elif name == "eta2":
            table = self._compute_estimator_2(table, globdat)
        return table

    def WRITEMESH(self, globdat, fname, ftype, **kwargs):
        if "manual" in ftype:
            self._write_mesh(globdat, fname)
        else:
            raise ValueError("Invalid file type passed to WRITEMESH")

    def configure(self, globdat, *, p, boundary):
        # get props
        check_dict(self, boundary, ["groups"])
        check_list(self, boundary["groups"])
        self._p = p
        self._bgroups = boundary["groups"]

        bnodes = set()
        for group in self._bgroups:
            ngroup = globdat[gn.NGROUPS][group]
            for inode in ngroup:
                bnodes.add(inode)
        self._bnodes = list(bnodes)

    def _perturb_nodes(self, nodes, globdat, meshsize, rng=np.random.default_rng()):
        h = np.max(meshsize[""])

        for inode, node in enumerate(nodes):
            if inode not in self._bnodes:
                alpha_i_bar = rng.uniform(-0.5, 0.5)
                ielem = inode - 1
                h_i_bar = min(meshsize[""][ielem], meshsize[""][ielem + 1])
                alpha_i = (h_i_bar / h) ** self._p * alpha_i_bar

                coords = node.get_coords()
                coords += h**self._p * alpha_i

                node.set_coords(coords)

        return nodes

    def _compute_estimator_1(self, table, globdat):
        xtable = to_xtable(table)
        jcol = xtable.add_column("eta1")

        elems = globdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = globdat[gn.DOFSPACE]

        size_col = globdat["elemTables"]["size"].get_column("")
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss1")
        h = np.max(globdat["elemTables"]["size"][""])

        for ielem, elem in enumerate(elems):
            inodes = elems.get_elem_nodes(ielem)
            idofs = dofs.get_dofs(inodes, dofs.get_types())
            coords = nodes.get_some_coords(inodes)
            grads, weights = shape.get_shape_gradients(coords)
            eldisp = globdat[gn.STATE0][idofs]

            expectation = 0

            def ref_grad_func(x):
                sgrads = shape.eval_global_shape_gradients([x], coords)
                return eldisp @ sgrads

            def ref_grad_prev(x):
                refnodes = elems.get_elem_nodes(ielem - 1)
                refdofs = dofs.get_dofs(refnodes, dofs.get_types())
                refcoords = nodes.get_some_coords(refnodes)
                refdisp = globdat[gn.STATE0][refdofs]

                sgrads = shape.eval_global_shape_gradients([x], refcoords)
                return refdisp @ sgrads

            def ref_grad_next(x):
                refnodes = elems.get_elem_nodes(ielem + 1)
                refdofs = dofs.get_dofs(refnodes, dofs.get_types())
                refcoords = nodes.get_some_coords(refnodes)
                refdisp = globdat[gn.STATE0][refdofs]

                sgrads = shape.eval_global_shape_gradients([x], refcoords)
                return refdisp @ sgrads

            for pglobdat in globdat["perturbedSolves"]:
                eldisp_p = pglobdat[gn.STATE0][idofs]
                nodes_p = pglobdat[gn.NSET]
                coords_p = nodes_p.get_some_coords(inodes)

                def pert_grad_func(x):
                    sgrads_p = shape.eval_global_shape_gradients([x], coords_p)
                    return eldisp_p @ sgrads_p

                def estimator_func(x):
                    return (ref_grad_func(x) - pert_grad_func(x)) ** 2

                def estimator_func_prev(x):
                    return (ref_grad_prev(x) - pert_grad_func(x)) ** 2

                def estimator_func_next(x):
                    return (ref_grad_next(x) - pert_grad_func(x)) ** 2

                norm = 0

                lp = coords_p[0, 0]
                rp = coords_p[0, 1]
                l = coords[0, 0]
                r = coords[0, 1]
                hp = h**self._p

                # If true, use the expressions from halfway through the proofs in Lemma 5.3
                midproof = True

                if lp >= l:
                    if rp > r:  # A++
                        if midproof:
                            a_r = (rp - r) / hp
                            j_r = self._grad_jump(ielem, 1, globdat)
                            I_i = hp**2 * (r - lp) / (rp - lp) ** 2 * a_r**2 * j_r**2
                            I_inext = (
                                hp * a_r * (hp * a_r / (rp - lp) - 1) ** 2 * j_r**2
                            )
                            norm += I_i
                            norm += I_inext
                        else:
                            norm += quad(estimator_func, lp, r)[0]
                            norm += quad(estimator_func_next, r, rp)[0]
                    else:  # A+-
                        if midproof:
                            pass
                        else:
                            norm += quad(estimator_func, lp, rp)[0]
                else:
                    if rp > r:  # A-+
                        if midproof:
                            a_l = (lp - l) / hp
                            a_r = (rp - r) / hp
                            j_l = self._grad_jump(ielem, -1, globdat)
                            j_r = self._grad_jump(ielem, 1, globdat)
                            xi_i = a_l * j_l + a_r * j_r
                            I_iprev = -hp * a_l * (hp / (rp - lp) * xi_i + j_l) ** 2
                            I_i = hp**2 * (r - l) / (rp - lp) ** 2 * xi_i**2
                            I_inext = hp * a_r * (hp / (rp - lp) * xi_i - j_r) ** 2
                            norm += I_iprev
                            norm += I_i
                            norm += I_inext
                        else:
                            norm += quad(estimator_func_prev, lp, l)[0]
                            norm += quad(estimator_func, l, r)[0]
                            norm += quad(estimator_func_next, r, rp)[0]
                    else:  # A--
                        if midproof:
                            a_l = (lp - l) / hp
                            j_l = self._grad_jump(ielem, -1, globdat)
                            I_iprev = (
                                -hp * a_l * (hp * a_l / (rp - lp) + 1) ** 2 * j_l**2
                            )
                            I_i = hp**2 * (rp - l) / (rp - lp) ** 2 * a_l**2 * j_l**2
                            norm += I_iprev
                            norm += I_i
                        else:
                            norm += quad(estimator_func_prev, lp, l)[0]
                            norm += quad(estimator_func, l, rp)[0]

                expectation += norm

            expectation /= len(globdat["perturbedSolves"])

            h = globdat["elemTables"]["size"].get_value(ielem, size_col)
            eta_2 = h ** -(self._p - 1) * expectation
            eta = np.sqrt(eta_2)

            xtable.set_value(ielem, jcol, eta)

        table = xtable.to_table()
        return table

    def _compute_estimator_2(self, table, globdat):
        xtable = to_xtable(table)
        jcol = xtable.add_column("eta2")

        elems = globdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = globdat[gn.DOFSPACE]

        size_col = globdat["elemTables"]["size"].get_column("")
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss1")
        h = np.max(globdat["elemTables"]["size"][""])

        for ielem, elem in enumerate(elems):
            inodes = elems.get_elem_nodes(ielem)
            idofs = dofs.get_dofs(inodes, dofs.get_types())
            coords = nodes.get_some_coords(inodes)
            grads, weights = shape.get_shape_gradients(coords)

            eldisp = globdat[gn.STATE0][idofs]
            strain = grads[:, :, 0].T @ eldisp
            norm_K = weights[0]

            expectation = 0

            for pglobdat in globdat["perturbedSolves"]:
                nodes_p = pglobdat[gn.NSET]
                coords_p = nodes_p.get_some_coords(inodes)
                grads_p, _ = shape.get_shape_gradients(coords_p)
                disp_p = pglobdat[gn.STATE0][idofs]
                strain_p = grads_p[:, :, 0].T @ disp_p

                norm = 0

                lp = coords_p[0, 0]
                rp = coords_p[0, 1]
                l = coords[0, 0]
                r = coords[0, 1]
                hp = h**self._p

                # If true, use the expressions from halfway through the proofs in Lemma 5.4
                midproof = True

                if lp >= l:
                    if rp > r:  # A++
                        if midproof:
                            j_r = self._grad_jump(ielem, 1, globdat)
                            a_r = (rp - r) / hp
                            norm += hp**2 * j_r**2 * a_r**2 / (rp - lp) ** 2
                        else:
                            norm += (strain - strain_p) ** 2
                    else:  # A+-
                        if midproof:
                            pass
                        else:
                            norm += (strain - strain_p) ** 2
                else:
                    if rp > r:  # A-+
                        if midproof:
                            j_l = self._grad_jump(ielem, -1, globdat)
                            j_r = self._grad_jump(ielem, 1, globdat)
                            a_l = (lp - l) / hp
                            a_r = (rp - r) / hp
                            xi_i = a_l * j_l + a_r * j_r
                            norm += hp**2 * xi_i**2 / (rp - lp) ** 2
                        else:
                            norm += (strain - strain_p) ** 2
                    else:  # A--
                        if midproof:
                            j_l = self._grad_jump(ielem, -1, globdat)
                            a_l = (lp - l) / hp
                            norm += hp**2 * j_l**2 * a_l**2 / (rp - lp) ** 2
                        else:
                            norm += (strain - strain_p) ** 2

                expectation += norm

            expectation /= len(globdat["perturbedSolves"])

            h_i = globdat["elemTables"]["size"].get_value(ielem, size_col)
            eta_2 = h_i ** -(2 * self._p - 2) * norm_K * expectation
            eta = np.sqrt(eta_2)

            xtable.set_value(ielem, jcol, eta)

        table = xtable.to_table()
        return table

    def _grad_jump(self, ielem, offset, globdat):
        assert abs(offset) == 1

        elems = globdat[gn.ESET]
        nodes = elems.get_nodes()
        dofs = globdat[gn.DOFSPACE]
        shape = globdat[gn.SHAPEFACTORY].get_shape(globdat[gn.MESHSHAPE], "Gauss1")

        def get_strain(ielem):
            inodes = elems.get_elem_nodes(ielem)
            idofs = dofs.get_dofs(inodes, dofs.get_types())
            coords = nodes.get_some_coords(inodes)
            grads, _ = shape.get_shape_gradients(coords)
            disp = globdat[gn.STATE0][idofs]
            strain = grads[:, :, 0].T @ disp
            return strain

        this_strain = get_strain(ielem)
        that_strain = get_strain(ielem + offset)

        if offset > 0:
            return that_strain - this_strain
        else:
            return this_strain - that_strain

    def _write_mesh(self, globdat, fname):
        nodes = globdat[gn.NSET]
        elems = globdat[gn.ESET]

        path, file = os.path.split(fname)
        if len(path) > 0 and not os.path.isdir(path):
            os.makedirs(path)

        with open(fname, "w") as file:
            file.write("nodes (ID, x, [y], [z])\n")
            for inode, node in enumerate(nodes):
                node_id = nodes.get_node_id(inode)
                coords = node.get_coords()
                file.write("{} ".format(node_id))
                file.write(" ".join(["{}".format(coord) for coord in coords]))
                file.write("\n")

            file.write("elements (node#1, node#2, [node#3, ...])\n")
            for ielem, elem in enumerate(elems):
                inodes = elems.get_elem_nodes(ielem)
                node_ids = nodes.get_node_ids(inodes)
                file.write(" ".join(["{}".format(node_id) for node_id in node_ids]))
                file.write("\n")

import numpy as np
from copy import deepcopy
from warnings import warn

from myjive.app import main
from myjive.fem import to_xnodeset
from myjive.names import GlobNames as gn
from myjive.util.proputils import split_off_type, check_dict, check_list
from myjivex import declare_all

from fem.meshing import read_mesh, write_mesh, calc_elem_sizes, get_patches_around_nodes


class RMFEMRunner:
    def __init__(
        self,
        *,
        inner,
        p,
        n_sample,
        update_type,
        globdat_keys=None,
        seed=None,
        run_modules=None,
        fname=None
    ):
        if update_type not in ["switch_file", "modify_file", "in_place"]:
            raise ValueError

        inner_cls, inner_kws = split_off_type(inner)
        self._p = p
        self._n_sample = n_sample
        self._update_type = update_type
        self._rng = np.random.default_rng(seed)
        self._globdat_keys = globdat_keys

        if inner_cls is main.jive:
            self._inner = inner_cls
            self._props = inner_kws
        else:
            self._inner = inner_cls
            self._props = inner_kws

        if self._update_type == "modify_file":
            self._fname = fname
        elif self._update_type == "in_place":
            if self._inner is main.jive:
                self._globdat_ref = main.jive(self._props, extra_declares=[declare_all])
                self._run_modules = run_modules
                check_list(self, run_modules)
            else:
                raise ValueError
        else:
            raise NotImplementedError

    def __call__(self):
        if self._update_type == "modify_file":
            nodes, elems = read_mesh(self._fname)
        elif self._update_type == "in_place":
            nodes = self._globdat_ref[gn.NSET]
            elems = self._globdat_ref[gn.ESET]
        else:
            raise NotImplementedError

        ref_elem_sizes = calc_elem_sizes(elems)
        ref_coords = np.copy(nodes.get_coords())

        samples = []

        for i in range(self._n_sample):
            new_coords = self._calc_perturbed_coords(ref_coords, elems, ref_elem_sizes)

            if self._update_type == "modify_file":
                to_xnodeset(nodes)
                nodes.set_coords(new_coords)
                write_mesh(elems, self._fname)
                nodes.to_nodeset()

                if self._inner is main.jive:
                    globdat = main.jive(self._props, extra_declares=[declare_all])
                    subset = dict((key, globdat[key]) for key in self._globdat_keys)
                    sample = subset
                else:
                    inner = self._inner(**self._props)
                    sample = inner()
                samples.append(sample)

            elif self._update_type == "in_place":
                assert self._inner is main.jive

                to_xnodeset(nodes)
                nodes.set_coords(new_coords)
                assert nodes == self._globdat_ref[gn.NSET]

                for name in self._run_modules:
                    module = self._globdat_ref[gn.MODULES][name]
                    module.run(self._globdat_ref)

                subset = dict(
                    (key, self._globdat_ref[key]) for key in self._globdat_keys
                )
                sample = deepcopy(subset)
                samples.append(sample)

            else:
                raise NotImplementedError

        to_xnodeset(nodes)
        nodes.set_coords(ref_coords)
        nodes.to_nodeset()

        if self._update_type == "modify_file":
            write_mesh(elems, self._fname)
        elif self._update_type == "in_place":
            assert nodes == self._globdat_ref[gn.NSET]
        else:
            raise NotImplementedError

        return samples

    def _calc_perturbed_coords(self, ref_coords, elems, elem_sizes):
        h = np.max(elem_sizes)
        rank = ref_coords.shape[1]
        pert_coords = np.copy(ref_coords)

        patches = get_patches_around_nodes(elems)

        for inode, coords in enumerate(pert_coords):
            # if inode in self._omit_nodes:
            #     continue

            if rank == 1:
                alpha_i_bar = np.array([self._rng.uniform(-0.5, 0.5)])
            elif rank == 2:
                r = 0.25 * np.sqrt(self._rng.uniform(0.0, 1.0))
                theta = self._rng.uniform(0.0, 2 * np.pi)
                alpha_i_bar = r * np.array([np.cos(theta), np.sin(theta)])
            else:
                raise NotImplementedError(
                    "RandomMeshModel has not been implemented for 3D yet"
                )

            patch = patches[inode]
            h_i_bar = np.min(elem_sizes[patch])
            alpha_i = (h_i_bar / h) ** self._p * alpha_i_bar

            warn("boundary conditions have not been implemented properly")
            # for group, dof in zip(self._bgroups, self._bdofs):
            #     ngroup = globdat[gn.NGROUPS][group]
            #     if inode in ngroup:
            #         if dof == "dx":
            #             alpha_i[0] = 0.0
            #         elif dof == "dy":
            #             alpha_i[1] = 0.0
            #         elif dof == "dz":
            #             alpha_i[2] = 0.0
            if inode in [0, ref_coords.shape[0] - 1]:
                alpha_i *= 0.0

            coords += h**self._p * alpha_i

            pert_coords[inode] = coords

        return pert_coords

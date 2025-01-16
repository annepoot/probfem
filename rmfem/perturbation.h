extern "C" {
  void calc_perturbed_coords
  (
    double* pert_coords_ptr,
    long* pert_coords_shape,
    double p,
    double tol,
    double* elem_sizes_ptr,
    long* elem_sizes_shape,
    double* std_uniform_ptr,
    long* std_uniform_shape,
    long** patches_patch_ptr,
    long* patches_patch_size,
    long patches_size,
    long* boundary_nodes_ptr,
    long* boundary_nodes_shape,
    double* boundary_normals_ptr,
    long* boundary_normals_shape,
    long* omit_nodes_ptr,
    long* omit_nodes_shape
  );
}

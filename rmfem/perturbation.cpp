#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "perturbation.h"

// compile with `g++ -shared -fPIC -o libperturbation.so perturbation.cpp`

double pi = 3.141592653589793238462643383279502884197169399375105820974944;

void calc_perturbed_coords(
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
    long* omit_nodes_shape )
{
  long node_count = pert_coords_shape[0];
  long rank = pert_coords_shape[1];
  double h = *std::max_element(elem_sizes_ptr, elem_sizes_ptr + elem_sizes_shape[0]);

  double alpha_i_bar[rank];
  double alpha_i[rank];
  double basis[rank];

  long boundary_node_count = boundary_normals_shape[0];
  long boundary_normal_count = boundary_normals_shape[1];
  long boundary_rank = boundary_normals_shape[2];

  if ( boundary_normal_count * boundary_rank != rank * rank ){
    throw;
  }

  long* boundary_nodes_start = boundary_nodes_ptr;
  long* boundary_nodes_end = boundary_nodes_ptr + boundary_nodes_shape[0];

  long* omit_nodes_start = omit_nodes_ptr;
  long* omit_nodes_end = omit_nodes_ptr + omit_nodes_shape[0];

  for ( long inode = 0; inode < node_count; inode++ ){
    if ( std::find(omit_nodes_start, omit_nodes_end, inode) != omit_nodes_end ){
      continue;
    }

    if ( rank == 1 ){
      alpha_i_bar[0] = std_uniform_ptr[inode] - 0.5;
    } else if ( rank == 2 ){
      double r = 0.25 * std::sqrt(std_uniform_ptr[rank * inode]);
      double theta = std_uniform_ptr[rank * inode + 1] * 2 * pi;
      alpha_i_bar[0] = r * std::cos(theta);
      alpha_i_bar[1] = r * std::sin(theta);
    } else {
      throw;
    }

    long* patch_ptr = patches_patch_ptr[inode];
    long patch_size = patches_patch_size[inode];
    double h_i_bar = h;

    for ( long ip = 0; ip < patch_size; ip++ ){
      double h_elem = elem_sizes_ptr[patch_ptr[ip]];
      if ( h_elem < h_i_bar ){
        h_i_bar = h_elem;
      }
    }

    for ( long ir = 0; ir < rank; ir++ ){
      alpha_i[ir] = std::pow(h_i_bar / h, p) * alpha_i_bar[ir];
    }

    long* boundary_idx = std::find(boundary_nodes_start, boundary_nodes_end, inode);
    if ( boundary_idx != boundary_nodes_end ){
      if ( rank == 1 ){
        alpha_i[0] *= 0.0;
      } else if ( rank == 2 ){
        long offset = boundary_idx - boundary_nodes_start;
        double norm1x = boundary_normals_ptr[(offset * 4) + 0];
        double norm1y = boundary_normals_ptr[(offset * 4) + 1];
        double norm2x = boundary_normals_ptr[(offset * 4) + 2];
        double norm2y = boundary_normals_ptr[(offset * 4) + 3];

        if ( std::abs(norm1x * norm2y - norm1y * norm2x) < tol ){
          basis[0] = -norm1y;
          basis[1] = norm2x;
          double weight = alpha_i[0] * basis[0] + alpha_i[1] * basis[1];
          alpha_i[0] = weight * basis[0];
          alpha_i[1] = weight * basis[1];
        } else {
          alpha_i[0] = 0.0;
          alpha_i[1] = 0.0;
        }
      } else {
        throw;
      }
    }

    for ( long ir = 0; ir < rank; ir++ ){
      pert_coords_ptr[inode * rank + ir] += std::pow(h, p) * alpha_i[ir];
    }
  }
}


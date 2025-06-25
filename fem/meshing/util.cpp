#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "util.h"

// compile with `g++ -shared -fPIC -o libutil.so util.cpp`
bool point_in_bbox( double* point, double* bbox, double tol ){
  if ( point[0] < bbox[0] - tol ){
    return false;
  } else if ( point[0] > bbox[2] + tol ){
    return false;
  } else if ( point[1] < bbox[1] - tol ){
    return false;
  } else if ( point[1] > bbox[3] + tol ){
    return false;
  } else {
    return true;
  }
}

long create_phi_matrix_t3(
  LONG_ARRAY_PTR phi_rowidx,
  LONG_ARRAY_PTR phi_colidx,
  DOUBLE_ARRAY_PTR phi_values,
  POINTSET_PTR coarse_nodes,
  GROUPSET_PTR coarse_elems,
  POINTSET_PTR fine_nodes,
  GROUPSET_PTR fine_elems
)
{
  double tol = 1e-8;

  int rank = 2;
  int elem_node_count = 3;

  int elem_count_c = coarse_elems.data.shape[0];
  int node_count_c = coarse_nodes.data.shape[0];
  int node_count_f = fine_nodes.data.shape[0];

  if ( rank != coarse_nodes.data.shape[1] || rank != fine_nodes.data.shape[1] ){
    throw;
  }

  if ( elem_node_count != coarse_elems.data.shape[1] || elem_node_count != fine_elems.data.shape[1] ){
    throw;
  }

  // each row is like (lbound1, ubound1, lbound2, ubound2)
  double* bboxes = new double[4 * elem_count_c];
  double coords[2 * elem_node_count];
  double bounds[4];
  int inodes_c[3];

  for ( int ielem_c = 0; ielem_c < elem_count_c; ielem_c++ ){
    if ( coarse_elems.sizes.ptr[ielem_c] != 3 ){
      throw;
    }

    for ( int in = 0; in < elem_node_count; in++ ){
      int inode_c = coarse_elems.data.ptr[elem_node_count * ielem_c + in];
      coords[rank * in] = coarse_nodes.data.ptr[rank * inode_c];
      coords[rank * in + 1] = coarse_nodes.data.ptr[rank * inode_c + 1];
    }

    bounds[0] = bounds[2] = coords[0];
    bounds[1] = bounds[3] = coords[1];

    for ( int in = 1; in < elem_node_count; in++ ){
      double xcoord = coords[in * rank];
      double ycoord = coords[in * rank + 1];

      if ( xcoord < bounds[0] ){
        bounds[0] = xcoord;
      } else if ( xcoord > bounds[2] ){
        bounds[2] = xcoord;
      }

      if ( ycoord < bounds[1] ){
        bounds[1] = ycoord;
      } else if ( ycoord > bounds[3] ){
        bounds[3] = ycoord;
      }

      for ( int i = 0; i < 4; i++ ){
        bboxes[4 * ielem_c + i] = bounds[i];
      }
    }
  }

  long phi_idx = 0;

  for ( int inode_f = 0; inode_f < node_count_f; inode_f++ ){
    for ( int ielem_c = 0; ielem_c < elem_count_c; ielem_c++ ){
      double* pp = fine_nodes.data.ptr + rank * inode_f;
      bool inside = point_in_bbox( pp , bboxes + 4 * ielem_c, tol );
      if ( inside ){
        for ( int in = 0; in < elem_node_count; in++ ){
          inodes_c[in] = coarse_elems.data.ptr[ elem_node_count * ielem_c + in ];
        }

        for ( int in = 0; in < elem_node_count; in++ ){
          coords[ rank * in ] = coarse_nodes.data.ptr[ rank * inodes_c[in] ];
          coords[ rank * in + 1 ] = coarse_nodes.data.ptr[ rank * inodes_c[in] + 1 ];
        }

        // solve p = a + (b - a) s + (c - a) t for s and t
        double pax = *pp - coords[0];
        double pay = *(pp + 1) - coords[1];
        double bax = coords[2] - coords[0];
        double bay = coords[3] - coords[1];
        double cax = coords[4] - coords[0];
        double cay = coords[5] - coords[1];

        double det = bax * cay - bay * cax;
        double s = ( cay * pax - cax * pay ) / det;
        double t = (-bay * pax + bax * pay ) / det;

        if ( s > -tol && s < 1.0 + tol && t > -tol && t < 1.0 + tol && s + t < 1.0 + tol ){
          if ( std::abs( 1 - s - t ) > tol ){
            phi_rowidx.ptr[phi_idx] = inode_f;
            phi_colidx.ptr[phi_idx] = inodes_c[0];
            phi_values.ptr[phi_idx] = 1 - s - t;
            phi_idx++;
          }

          if ( std::abs(s) > tol ){
            phi_rowidx.ptr[phi_idx] = inode_f;
            phi_colidx.ptr[phi_idx] = inodes_c[1];
            phi_values.ptr[phi_idx] = s;
            phi_idx++;
          }

          if ( std::abs(t) > tol ){
            phi_rowidx.ptr[phi_idx] = inode_f;
            phi_colidx.ptr[phi_idx] = inodes_c[2];
            phi_values.ptr[phi_idx] = t;
            phi_idx++;
          }
        }
      }
    }
  }

  delete[] bboxes;

  return phi_idx;
}


double dist (
  double p1x,
  double p1y,
  double p2x,
  double p2y,
  double p3x,
  double p3y
)
{
  return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
}

bool check_point_in_polygon(
    double* point,
    double* polygon,
    int size,
    double tol
 )
{
  double min = 0.0;
  double max = 0.0;

  double px = point[0];
  double py = point[1];

  for ( int i = 0; i < size; i++ ){
    int j = (i + 1) % size;

    double ax = polygon[2 * i];
    double ay = polygon[2 * i + 1];
    double bx = polygon[2 * j];
    double by = polygon[2 * j + 1];

    double d = dist(px, py, ax, ay, bx, by);

    if ( d < min - tol){
      if ( max > tol ){
        return false;
      }
      min = d;
    } else if ( d > max + tol){
      if ( min < -tol ){
        return false;
      }
      max = d;
    }
  }

  return true;
}


#ifndef CTYPES_UTILS_H
#define CTYPES_UTILS_H

#include <jem/util/Properties.h>

using jem::util::Properties;

#ifdef __cplusplus
extern "C" {
#endif

struct LONG_ARRAY_PTR{
  long* ptr;
  long* shape;
  long dim;
};

struct DOUBLE_ARRAY_PTR{
  double* ptr;
  long* shape;
  long dim;
};

struct SPARSE_MAT_PTR{
  DOUBLE_ARRAY_PTR values;
  LONG_ARRAY_PTR indices;
  LONG_ARRAY_PTR offsets;
};

struct SHAPE_PTR{
  char* type;
  char* ischeme;
};

struct CONSTRAINTS_PTR{
  LONG_ARRAY_PTR dofs;
  DOUBLE_ARRAY_PTR values;
};

struct POINTSET_PTR{
  DOUBLE_ARRAY_PTR data;
};

struct GROUPSET_PTR{
  LONG_ARRAY_PTR data;
  LONG_ARRAY_PTR sizes;
};

struct DOFSPACE_PTR{
  LONG_ARRAY_PTR data;
};

struct GLOBDAT {
  POINTSET_PTR nodeSet;
  GROUPSET_PTR elementSet;
  DOFSPACE_PTR dofSpace;
  DOUBLE_ARRAY_PTR state0;
  DOUBLE_ARRAY_PTR intForce;
  DOUBLE_ARRAY_PTR extForce;
  SPARSE_MAT_PTR matrix0;
  CONSTRAINTS_PTR constraints;
  SHAPE_PTR shape;
};

void globdatToCtypes
	( GLOBDAT& iodat,
	  Properties globdat,
	  long flags );

#ifdef __cplusplus
}
#endif

#endif
extern "C" {
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

  struct SHAPE_PTR{
    char* type;
    char* ischeme;
  };

  struct POINTSET_PTR{
    DOUBLE_ARRAY_PTR data;
  };

  struct GROUPSET_PTR{
    LONG_ARRAY_PTR data;
    LONG_ARRAY_PTR sizes;
  };

  void create_phi_matrix_t3
  (
    DOUBLE_ARRAY_PTR phi,
    POINTSET_PTR coarse_nodes,
    GROUPSET_PTR coarse_elems,
    POINTSET_PTR fine_nodes,
    GROUPSET_PTR fine_elems
  );

  bool check_point_in_polygon
  (
    double* point,
    double* polygon,
    int vertex_count,
    double tol
  );
}

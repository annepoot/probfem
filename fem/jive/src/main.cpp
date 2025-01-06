#include <jem/base/Throwable.h>

#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/InfoModule.h>
#include <jive/app/SampleModule.h>
#include <jive/app/UserconfModule.h>
#include <jive/geom/declare.h>
#include <jive/model/declare.h>
#include <jive/femodel/declare.h>
#include <jive/fem/declare.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/implict/declare.h>
#include <jive/app/declare.h>
#include <jive/algebra/declare.h>
#include <jive/gl/declare.h>

#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/model/Names.h>
#include <jive/model/StateVector.h>
#include <jive/fem/NodeSet.h>

#include "models.h"
#include "modules.h"
#include "materials.h"
#include "ExposedApplication.h"

using namespace jem;

using jive::app::Module;
using jive::app::ChainModule;
using jive::app::OutputModule;
using jive::app::InfoModule;
using jive::app::ControlModule;
using jive::app::ReportModule;
using jive::app::SampleModule;
using jive::app::UserconfModule;
using jive::fem::InputModule;
using jive::fem::InitModule;
using jive::fem::ShapeModule;

using jive::algebra::AbstractMatrix;
using jive::algebra::SparseMatrixObject;
using jive::algebra::MatrixBuilder;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::IntMatrix;
using jive::util::DofSpace;
using jive::util::Constraints;
using jive::model::STATE0;
using jive::model::PropertyNames;
using jive::model::StateVector;
using jive::fem::NodeSet;


//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------


Ref<Module> mainModule ()
{
  Ref<ChainModule>    chain = newInstance<ChainModule> ();

  // Declare internal shapes, models and matrix builders. These
  // functions essentially store pointers to construction functions
  // that are called when Jive needs to create a shape, model or
  // matrix builder of a particular type.

  jive::geom:: declareIShapes     ();
  jive::geom::declareShapes       ();
  jive::model::declareModels      ();
  jive::fem::declareMBuilders     ();
  jive::implict::declareModels    ();
  jive::algebra::declareMBuilders ();
  jive::femodel::declareModels    ();

  declareModels                   ();

  // Declare all modules that you want to add dynamically.

  jive::app     ::declareModules  ();
  jive::implict ::declareModules  ();
  jive::gl      ::declareModules  ();

  declareModules                  ();

  // Declare all materials that you want to add dynamically.

  declareMaterials                ();

  // Set up the module chain. These modules will be called by Jive in
  // the order that they have been added to the chain.

  // User defined inputmodules

  chain->pushBack ( newInstance<UserconfModule> ( "userinput" ) );

  // The ShapeModule creates a ShapeTable that maps elements to shape
  // objects. This is needed, among others, by the FemViewModule
  // below.

  chain->pushBack ( newInstance<ShapeModule>() );

  // The InitModule creates the main model and initializes some other
  // stuff.

  chain->pushBack ( newInstance<InitModule>() );

  // The InfoModule prints information about the current calculation.

  chain->pushBack ( newInstance<InfoModule>() );

  // UserModules to specify solver and output

  chain->pushBack ( newInstance<UserconfModule> ( "usermodules" ) );

  // The SampleModule is used to generate simple numeric ouput

  chain->pushBack ( newInstance<SampleModule> ( ) );

  chain->pushBack ( newInstance<ControlModule> ( "control" ) );

  // Finally, the chain is wrapped in a ReportModule that prints some
  // overall information about the current calculation.

  return newInstance<ReportModule> ( "report", chain );
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


#ifdef __cplusplus
extern "C" {
#endif

struct INT_VEC_PTR{
  int* ptr;
  int size;
};

struct DOUBLE_VEC_PTR{
  double* ptr;
  int size;
};

struct INT_MAT_PTR{
  int* ptr;
  int size0;
  int size1;
};

struct DOUBLE_MAT_PTR{
  double* ptr;
  int size0;
  int size1;
};

struct SPARSE_MAT_PTR{
  DOUBLE_VEC_PTR values;
  INT_VEC_PTR indices;
  INT_VEC_PTR offsets;
};

struct CONSTRAINTS_PTR{
  INT_VEC_PTR dofs;
  DOUBLE_VEC_PTR values;
};

struct GLOBDAT {
  DOUBLE_VEC_PTR state0;
  SPARSE_MAT_PTR matrix0;
  DOUBLE_MAT_PTR coords;
  INT_MAT_PTR dofs;
  CONSTRAINTS_PTR constraints;
};

void getGlobdat
	( GLOBDAT& outdat,
	  char* fname )
{
  int argc = 2;
  char *argv[] = { (char*)"name", fname, NULL };

  Properties globdat ( "globdat" );

  ExposedApplication::exec ( argc, argv, & mainModule, globdat );

  NodeSet nodes = NodeSet::get( globdat, "" );
  Ref<DofSpace> dofs = DofSpace::get ( globdat, "" );
  Ref<Constraints> cons = Constraints::get ( dofs, globdat );

  idx_t nodeCount = nodes.size();
  idx_t rank = nodes.rank();
  idx_t typeCount = dofs->typeCount();
  idx_t dofCount = dofs->dofCount();

  Vector u ( dofs->dofCount() );
  StateVector::get ( u, STATE0, dofs, globdat );
  Matrix coords( nodes.rank(), nodes.size() );
  nodes.getCoords(coords);

  // Populate state0 array
  if ( dofCount > outdat.state0.size ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t idof = 0; idof < dofCount ; idof++ ){
    outdat.state0.ptr[idof] = u[idof];
  }
  outdat.state0.size = dofCount;

  // Populate coords array
  if ( nodeCount > outdat.coords.size0 || rank > outdat.coords.size1 ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t inode = 0; inode < nodeCount ; inode++ ){
    for ( idx_t irank = 0; irank < rank ; irank++ ){
      outdat.coords.ptr[(inode * rank) + irank] = coords[inode][irank];
    }
  }
  outdat.coords.size0 = nodeCount;
  outdat.coords.size1 = rank;

  // Populate dof_idx array
  if ( nodeCount > outdat.dofs.size0 || typeCount > outdat.dofs.size1 ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( int inode = 0; inode < nodeCount; inode++ ){
	for ( int itype = 0; itype < typeCount; itype++ ){
	  outdat.dofs.ptr[(inode * typeCount) + itype] = dofs->getDofIndex(inode, itype);
	}
  }
  outdat.dofs.size0 = nodeCount;
  outdat.dofs.size1 = typeCount;

  // Populate matrix0 array
  Ref<MatrixBuilder> mbuilder;
  globdat.get( mbuilder, PropertyNames::MATRIX0 );

  Ref<AbstractMatrix> mat = mbuilder->getMatrix();
  Ref<SparseMatrixObject> smat = dynamicCast<SparseMatrixObject> ( mat );

  Vector values = smat->getValues();
  IdxVector colIndices = smat->getColumnIndices();
  IdxVector rowOffsets = smat->getRowOffsets();

  idx_t indexCount = colIndices.size();
  idx_t offsetCount = rowOffsets.size();

  if ( offsetCount != dofCount + 1 ){
    throw Exception( "getState0()", "matrix0 and state0 have incompatible sizes" );
  }

  if ( indexCount > outdat.matrix0.values.size ||
       indexCount > outdat.matrix0.indices.size ||
       offsetCount > outdat.matrix0.offsets.size ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t i = 0; i < indexCount ; i++ ){
    outdat.matrix0.values.ptr[i] = values[i];
    outdat.matrix0.indices.ptr[i] = colIndices[i];
  }

  for ( idx_t i = 0; i < offsetCount ; i++ ){
    outdat.matrix0.offsets.ptr[i] = rowOffsets[i];
  }

  outdat.matrix0.values.size = indexCount;
  outdat.matrix0.indices.size = indexCount;
  outdat.matrix0.offsets.size = offsetCount;

  // Populate constraints array
  idx_t cdofCount = cons->slaveDofCount();

  if ( cons->masterDofCount() > 0 ){
    throw Exception ( "getState0()", "master dofs have not been implemented");
  }

  if ( cdofCount > outdat.constraints.dofs.size ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  IdxVector cdofs = cons->getSlaveDofs();
  Vector cvals;
  cons->getRvalues(cvals, cdofs);

  for ( idx_t idof = 0; idof < cdofCount ; idof++ ){
    outdat.constraints.dofs.ptr[idof] = cdofs[idof];
  }
  outdat.constraints.dofs.size = cdofCount;
  outdat.constraints.values.size = cdofCount;
}


int run ( char* fname )
{
  int argc = 2;
  char *argv[] = { (char*)"name", fname, NULL };

  Properties globdat ( "globdat" );

  return ExposedApplication::exec ( argc, argv, & mainModule, globdat );
}

int exec ( int argc, char** argv )
{
  Properties globdat ( "globdat" );

  return ExposedApplication::exec ( argc, argv, & mainModule, globdat );
}

#ifdef __cplusplus
}
#endif

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
#include <jive/geom/Names.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/util/utilities.h>

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

using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::IntMatrix;
using jive::algebra::AbstractMatrix;
using jive::algebra::SparseMatrixObject;
using jive::algebra::MatrixBuilder;
using jive::geom::PropertyNames;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::model::STATE0;
using jive::model::ActionParams;
using jive::model::StateVector;
using jive::util::XDofSpace;
using jive::util::Constraints;
using jive::util::joinNames;



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

struct CHAR_ARRAY_PTR{
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
  GROUPSET_PTR elemSet;
  DOFSPACE_PTR dofSpace;
  DOUBLE_ARRAY_PTR state0;
  DOUBLE_ARRAY_PTR intForce;
  DOUBLE_ARRAY_PTR extForce;
  SPARSE_MAT_PTR matrix0;
  CONSTRAINTS_PTR constraints;
  SHAPE_PTR shape;
};

void getGlobdat
	( GLOBDAT& outdat,
	  char* fname )
{
  int argc = 2;
  char *argv[] = { (char*)"name", fname, NULL };

  Properties globdat ( "globdat" );

  ExposedApplication::exec ( argc, argv, & mainModule, globdat );

  ElementSet elems = ElementSet::get( globdat, "" );
  NodeSet nodes = elems.getNodes();
  Ref<XDofSpace> dofs = XDofSpace::get ( nodes.getData(), globdat );
  Ref<Constraints> cons = Constraints::get ( dofs, globdat );

  idx_t nodeCount = nodes.size();
  idx_t rank = nodes.rank();
  idx_t elemCount = elems.size();
  idx_t maxElemNodeCount = elems.maxElemNodeCount();
  idx_t typeCount = dofs->typeCount();
  idx_t dofCount = dofs->dofCount();

  Vector u ( dofs->dofCount() );
  Vector fint;
  Vector fext;

  StateVector::get ( u, STATE0, dofs, globdat );
  globdat.get( fint, ActionParams::INT_VECTOR );
  globdat.get( fext, ActionParams::EXT_VECTOR );

  // Populate state0 array
  if ( dofCount > outdat.state0.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t idof = 0; idof < dofCount ; idof++ ){
    outdat.state0.ptr[idof] = u[idof];
  }
  outdat.state0.shape[0] = dofCount;

  // Populate intForce array
  if ( dofCount > outdat.intForce.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t idof = 0; idof < dofCount ; idof++ ){
    outdat.intForce.ptr[idof] = fint[idof];
  }
  outdat.intForce.shape[0] = dofCount;

  // Populate extForce array
  if ( dofCount > outdat.extForce.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t idof = 0; idof < dofCount ; idof++ ){
    outdat.extForce.ptr[idof] = fext[idof];
  }
  outdat.extForce.shape[0] = dofCount;

  // Populate nodeSet
  if ( nodeCount > outdat.nodeSet.data.shape[0] || rank > outdat.nodeSet.data.shape[1] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  Vector coords (rank);
  for ( idx_t inode = 0; inode < nodeCount ; inode++ ){
    nodes.getNodeCoords(coords, inode);

    for ( idx_t ir = 0; ir < rank ; ir++ ){
      outdat.nodeSet.data.ptr[(inode * rank) + ir] = coords[ir];
    }
  }
  outdat.nodeSet.data.shape[0] = nodeCount;
  outdat.nodeSet.data.shape[1] = rank;

  // Populate elemSet
  if ( elemCount > outdat.elemSet.data.shape[0] ||
       maxElemNodeCount > outdat.elemSet.data.shape[1] ||
       elemCount > outdat.elemSet.sizes.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  IdxVector inodes (maxElemNodeCount);
  for ( idx_t ielem = 0; ielem < elemCount; ielem++ ){
    idx_t elemNodeCount = elems.getElemNodeCount(ielem);
    elems.getElemNodes(inodes, ielem);

    outdat.elemSet.sizes.ptr[ielem] = elemNodeCount;
    for ( idx_t in = 0; in < elemNodeCount; in++ ){
      outdat.elemSet.data.ptr[(ielem * maxElemNodeCount) + in] = inodes[in];
    }
  }
  outdat.elemSet.data.shape[0] = elemCount;
  outdat.elemSet.data.shape[1] = maxElemNodeCount;
  outdat.elemSet.sizes.shape[0] = elemCount;

  // Populate dofSpace array
  if ( nodeCount > outdat.dofSpace.data.shape[0] || typeCount > outdat.dofSpace.data.shape[1] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( int inode = 0; inode < nodeCount; inode++ ){
	for ( int itype = 0; itype < typeCount; itype++ ){
	  outdat.dofSpace.data.ptr[(inode * typeCount) + itype] = dofs->getDofIndex(inode, itype);
	}
  }
  outdat.dofSpace.data.shape[0] = nodeCount;
  outdat.dofSpace.data.shape[1] = typeCount;

  // Populate matrix0 array
  Ref<MatrixBuilder> mbuilder;
  globdat.get( mbuilder, ActionParams::MATRIX0 );

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

  if ( indexCount > outdat.matrix0.values.shape[0] ||
       indexCount > outdat.matrix0.indices.shape[0] ||
       offsetCount > outdat.matrix0.offsets.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t i = 0; i < indexCount ; i++ ){
    outdat.matrix0.values.ptr[i] = values[i];
    outdat.matrix0.indices.ptr[i] = colIndices[i];
  }

  for ( idx_t i = 0; i < offsetCount ; i++ ){
    outdat.matrix0.offsets.ptr[i] = rowOffsets[i];
  }

  outdat.matrix0.values.shape[0] = indexCount;
  outdat.matrix0.indices.shape[0] = indexCount;
  outdat.matrix0.offsets.shape[0] = offsetCount;

  // Populate constraints array
  idx_t cdofCount = cons->slaveDofCount();

  if ( cons->masterDofCount() > 0 ){
    throw Exception ( "getState0()", "master dofs have not been implemented");
  }

  if ( cdofCount > outdat.constraints.dofs.shape[0] ){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  IdxVector cdofs = cons->getSlaveDofs();
  Vector cvals ( cdofs.size() );
  cons->getRvalues(cvals, cdofs);

  for ( idx_t idof = 0; idof < cdofCount ; idof++ ){
    outdat.constraints.dofs.ptr[idof] = cdofs[idof];
    outdat.constraints.values.ptr[idof] = cvals[idof];
  }
  outdat.constraints.dofs.shape[0] = cdofCount;
  outdat.constraints.values.shape[0] = cdofCount;

  // Populate shape string
  String shapeType;
  String shapeScheme;

  globdat.get(shapeType, joinNames(PropertyNames::SHAPE, PropertyNames::TYPE));
  globdat.get(shapeScheme, joinNames(PropertyNames::SHAPE, PropertyNames::ISCHEME));

  int shapeTypeSize = shapeType.size();
  int shapeSchemeSize = shapeScheme.size();

  for ( idx_t is = 0; is < shapeTypeSize; is++ ){
    outdat.shape.type[is] = shapeType[is];
  }
  for ( idx_t is = 0; is < shapeSchemeSize; is++ ){
    outdat.shape.ischeme[is] = shapeScheme[is];
  }
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

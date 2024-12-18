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

#include <jive/util/DofSpace.h>
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

using jive::Vector;
using jive::Matrix;
using jive::IntMatrix;
using jive::util::DofSpace;
using jive::model::STATE0;
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

void getState0 
	( double* state0_ptr,
	  int* state0_size,
	  double* coords_ptr,
	  int* coords_size,
	  int* coords_rank,
	  int* dofs_ptr,
	  int* dofs_size,
	  int* dofs_rank,
	  char* fname )
{
  int argc = 2;
  char *argv[] = { "name", fname, NULL };

  Properties globdat ( "globdat" );

  ExposedApplication::exec ( argc, argv, & mainModule, globdat );

  NodeSet nodes = NodeSet::get( globdat, "" );
  Ref<DofSpace> dofs = DofSpace::get ( globdat, "" );

  idx_t nodeCount = nodes.size();
  idx_t rank = nodes.rank();
  idx_t typeCount = dofs->typeCount();
  idx_t dofCount = dofs->dofCount();

  Vector u ( dofs->dofCount() );
  StateVector::get ( u, STATE0, dofs, globdat );
  Matrix coords( nodes.rank(), nodes.size() );
  nodes.getCoords(coords);

  // Populate state0 array
  if (dofCount > *state0_size){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t idof = 0; idof < dofCount ; idof++ ){
    state0_ptr[idof] = u[idof];
  }
  *state0_size = dofCount;

  // Populate coords array
  if (nodeCount > *coords_size || rank > *coords_rank){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( idx_t inode = 0; inode < nodeCount ; inode++ ){
    for ( idx_t irank = 0; irank < rank ; irank++ ){
      coords_ptr[(inode * rank) + irank] = coords[inode][irank];
    }
  }
  *coords_rank = rank;
  *coords_size = nodeCount;

  // Populate dof_idx array
  if (nodeCount > *dofs_size || typeCount > *dofs_rank){
    throw Exception ( "getState0()", "buffer size insufficient");
  }

  for ( int inode = 0; inode < nodeCount; inode++ ){
	for ( int itype = 0; itype < typeCount; itype++ ){
	  dofs_ptr[(inode * typeCount) + itype] = dofs->getDofIndex(inode, itype);
	}
  }
  *dofs_rank = typeCount;
  *dofs_size = nodeCount;
}


int run ( char* fname )
{
  int argc = 2;
  char *argv[] = { "name", fname, NULL };

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

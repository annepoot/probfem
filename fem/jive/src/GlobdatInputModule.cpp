#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/Error.h>
#include <jem/base/ObjectTraits.h>
#include <jem/base/System.h>
#include <jem/numeric/utilities.h>
#include <jem/util/SparseArray.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/app/ModuleFactory.h>

#include "GlobdatInputModule.h"
#include "CtypesUtils.h"

using jem::io::endl;
using jem::util::SparseArray;
using jem::util::ArrayBuffer;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::newXElementSet;
using jive::fem::newXNodeSet;
using jive::fem::toXElementSet;
using jive::fem::toXNodeSet;

//=======================================================================
//   class GlobdatInputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* GlobdatInputModule::TYPE_NAME       = "GlobdatInput";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GlobdatInputModule::GlobdatInputModule

  ( const String&  name ) :

      Super   ( name   )

{
  rank_         = 0;
  numNodes_     = 0;
  numElems_     = 0;
}

GlobdatInputModule::~GlobdatInputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GlobdatInputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  configure_ ( conf, props, globdat );

  System::out() << "Reading input with " << myName_ << "\n";

  initSets_ ( globdat );

  readMesh_ ( globdat );

  return DONE;
}

//-----------------------------------------------------------------------
//   configure_
//-----------------------------------------------------------------------


void GlobdatInputModule::configure_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );

  // 2D only (for 3D: combine with LaminateMeshModule)
  rank_ = 2;
}

//-----------------------------------------------------------------------
//   initSets_
//-----------------------------------------------------------------------

void GlobdatInputModule::initSets_

  ( const Properties&   globdat )

{
  // find or create NodeSet

  nodes_ = toXNodeSet ( NodeSet::find ( globdat ) );

  if ( nodes_ == NIL )
  {
    // default name and storage mode!

    nodes_ = newXNodeSet ( );
  }

  nodes_.clear();

  // find or create ElementSet

  elems_ = toXElementSet ( ElementSet::find ( globdat ) );

  if ( elems_ == NIL )
  {
    elems_ = newXElementSet ( nodes_ );
  }

  elems_.clear();
}

//-----------------------------------------------------------------------
//   readMesh_
//-----------------------------------------------------------------------

void GlobdatInputModule::readMesh_

  ( const Properties&   globdat )

{
  POINTSET_PTR inputNodes = ObjectTraits<POINTSET_PTR>::toValue(globdat.get("input.nodeSet"));
  GROUPSET_PTR inputElems = ObjectTraits<GROUPSET_PTR>::toValue(globdat.get("input.elementSet"));

  if ( inputNodes.data.dim != 2 || inputElems.data.dim != 2 || inputElems.sizes.dim != 1 ){
    throw Error ( JEM_FUNC, "dimensionality mismatch" );
  }
  if ( inputElems.data.shape[0] != inputElems.sizes.shape[0] ){
    throw Error ( JEM_FUNC, "dimension mismatch" );
  }

  numNodes_ = inputNodes.data.shape[0];
  rank_ = inputNodes.data.shape[1];
  numElems_ = inputElems.data.shape[0];
  maxElemNodeCount_ = inputElems.data.shape[1];

  Vector     coords( rank_ );
  IdxVector  inodes;

  nodes_.reserve( numNodes_ );
  System::out() << " ...Adding " << numNodes_ << " nodes.\n";

  // read and store the nodes
  for ( idx_t in = 0; in < numNodes_; ++in)
  {
    for ( idx_t ir = 0; ir < rank_; ir++ ){
      coords[ir] = inputNodes.data.ptr[in * rank_ + ir];
    }
    nodes_.addNode( coords );
  }
  nodes_.store( globdat );

  // read and store the elements
  elems_.reserve( numElems_ );
  System::out() << " ...Adding " << numElems_ << " elements.\n";

  for ( idx_t ie = 0; ie < numElems_; ++ie)
  {
    idx_t nn = inputElems.sizes.ptr[ie];
    inodes.resize( nn );
    for ( idx_t in = 0; in < nn; ++ in )
    {
      inodes[in] = inputElems.data.ptr[ie * maxElemNodeCount_ + in];
    }
    elems_.addElement ( inodes );
  }
  elems_.store( globdat );
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  GlobdatInputModule::makeNew

  ( const String&           name,
    const Properties&       conf,
    const Properties&       props,
    const Properties&       globdat )

{
  return newInstance<Self> ( name );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareGlobdatInputModule
//-----------------------------------------------------------------------

void declareGlobdatInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( GlobdatInputModule::TYPE_NAME,
                         & GlobdatInputModule::makeNew );
}

#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/array/tensor.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/util/StringUtils.h>

#include <jive/geom/IShapeFactory.h>
#include <jive/geom/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/utilities.h>

#include "SpringModel.h"

using jem::numeric::MatmulChain;
using jive::geom::IShapeFactory;
using jive::util::joinNames;
using jive::StringVector;
using jem::util::StringUtils;
using jive::model::StateVector;
using jive::Cubix;

typedef MatmulChain<double,3>   MChain3;
typedef MatmulChain<double,2>   MChain2;
typedef MatmulChain<double,1>   MChain1;

//======================================================================
//   definition
//======================================================================

const char* SpringModel::DOF_NAMES[3]     = {"dx","dy","dz"};
const char* SpringModel::K_PROP           = "k";
const char* SpringModel::SHAPE_PROP       = "shape";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

SpringModel::SpringModel

   ( const String&       name,
     const Properties&   conf,
     const Properties&   props,
     const Properties&   globdat ) : Super(name)
{
  using jive::geom::PropertyNames;

  // create myTag_ (last part of myName_)
  StringVector names ( StringUtils::split( myName_, '.' ) );
  myTag_     = names [ names.size() - 1 ];

  Properties  myProps = props.getProps ( myName_ );
  Properties  myConf  = conf.makeProps ( myName_ );

  const String context = getContext();

  egroup_ = ElemGroup::get ( myConf, myProps, globdat, context );

  numElem_   = egroup_.size();
  ielems_    . resize( numElem_ );
  ielems_    = egroup_.getIndices ();
  elems_     = egroup_.getElements ( );
  nodes_     = elems_.getNodes     ( );
  rank_      = nodes_.rank         ( );
  numNode_   = nodes_.size         ( );

  // Make sure that the number of spatial dimensions (the rank of the
  // mesh) is valid.

  if ( rank_ < 1 || rank_ > 3 )
  {
    throw IllegalInputException (
      context,
      String::format (
        "invalid node rank: %d (should be 1, 2 or 3)", rank_
      )
    );
  }

  String shapeProp = joinNames ( myName_, SHAPE_PROP );

  String shapeType;
  String shapeScheme;
  props.getProps(shapeProp).get(shapeType, PropertyNames::TYPE);
  props.getProps(shapeProp).get(shapeScheme, PropertyNames::ISCHEME);

  globdat.set(joinNames(PropertyNames::SHAPE, PropertyNames::TYPE), shapeType);
  globdat.set(joinNames(PropertyNames::SHAPE, PropertyNames::ISCHEME), shapeScheme);

  shape_  = IShapeFactory::newInstance ( shapeProp, conf, props );

  nodeCount_  = shape_->nodeCount   ();
  ipCount_    = shape_->ipointCount ();
  dofCount_   = rank_ * nodeCount_;

  // Make sure that the rank of the shape matches the rank of the
  // mesh.

  if ( shape_->globalRank() != rank_ )
  {
    throw IllegalInputException (
      context,
      String::format (
        "shape has invalid rank: %d (should be %d)",
        shape_->globalRank (),
        rank_
      )
    );
  }

  // Make sure that each element has the same number of nodes as the
  // shape object.

  elems_.checkSomeElements (
    context,
    ielems_,
    shape_->nodeCount  ()
  );

  dofs_ = XDofSpace::get ( nodes_.getData(), globdat );

  dofTypes_.resize( rank_ );

  for( idx_t i = 0; i < rank_; i++)
  {
    dofTypes_[i] = dofs_->addType ( DOF_NAMES[i]);
  }

  dofs_->addDofs (
    elems_.getUniqueNodesOf ( ielems_ ),
    dofTypes_
  );

  IdxVector inodes ( nodeCount_ );
  Matrix coords ( rank_, nodeCount_ );
  Matrix ipCoords ( rank_, shape_->ipointCount() );

  getShapeFuncs_ = getShapeFuncsFunc ( rank_ );
}

SpringModel::~SpringModel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void SpringModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties  myProps = props.getProps ( myName_ );
  myProps.get  ( k_, K_PROP );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SpringModel::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties  myConf  = conf.makeProps ( myName_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool SpringModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( action == Actions::GET_MATRIX0 )
  {
    Ref<MatrixBuilder>  mbuilder;
    params.find( mbuilder, ActionParams::MATRIX0 );
    getMatrix_ ( mbuilder );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   getMatrix_
//-----------------------------------------------------------------------


void SpringModel::getMatrix_

  ( Ref<MatrixBuilder>  mbuilder ) const

{
  Matrix      coords     ( rank_, nodeCount_ );
  Matrix      elemMat    ( dofCount_, dofCount_ );

  Matrix      R          ( rank_, rank_ );

  Matrix      sfuncs     = shape_->getShapeFunctions ();
  Matrix      N          ( rank_, rank_ * nodeCount_ );
  Matrix      Nt         = N.transpose ( );

  IdxVector   inodes     ( nodeCount_ );
  IdxVector   idofs      ( dofCount_  );

  Vector      ipWeights  ( ipCount_   );

  MChain3     mc3;

  R = 0.0;

  for ( idx_t i = 0; i < rank_ ; i++ )
  {
    R(i,i) = k_;
  }

  // Iterate over all elements assigned to this model.
  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    // Get the global element index.
    idx_t  ielem = ielems_[ie];

    // Get the element coordinates and DOFs.
    elems_.getElemNodes  ( inodes, ielem    );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, dofTypes_ );

    // Assemble the element matrix and the internal force vector.
    elemMat   = 0.0;

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {
      // compute matrix of shape function N
      getShapeFuncs_ ( N, sfuncs(ALL,ip) );

      // Add the contribution of this integration point.
      elemMat   += ipWeights[ip] * mc3.matmul ( Nt, R, N );
    }

    // Add the element secant matrix to the global stiffness matrix.
    mbuilder->addBlock ( idofs, idofs, elemMat );
  }
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newSpringModel
//-----------------------------------------------------------------------


static Ref<Model>     newSpringModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<SpringModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declareSpringModel
//-----------------------------------------------------------------------


void declareSpringModel ()
{
  using jive::model::ModelFactory;
  ModelFactory::declare ( "Spring", & newSpringModel );
}

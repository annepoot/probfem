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

#include "LaplaceModel.h"

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

const char* LaplaceModel::DOF_NAMES[3]     = {"u"};
const char* LaplaceModel::SHAPE_PROP       = "shape";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

LaplaceModel::LaplaceModel

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
  dofType_ = dofs_->addType ( DOF_NAMES[0]);

  dofs_->addDofs (
    elems_.getUniqueNodesOf ( ielems_ ),
    dofType_
  );

  idx_t  ipCount = shape_->ipointCount() * egroup_.size();

  idx_t ipoint = 0;
  IdxVector inodes ( nodeCount_ );
  Matrix coords ( rank_, nodeCount_ );
  Matrix ipCoords ( rank_, shape_->ipointCount() );

  Matrix allIpCoords ( rank_, ipCount );

  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    idx_t  ielem = ielems_[ie];
    elems_.getElemNodes  ( inodes, ielem    );
    nodes_.getSomeCoords ( coords, inodes );
    shape_->getGlobalIntegrationPoints (ipCoords, coords );

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {
      allIpCoords[ipoint] = ipCoords[ip];
      ipoint++;
    }
  }
}

LaplaceModel::~LaplaceModel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void LaplaceModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LaplaceModel::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties  myConf  = conf.makeProps ( myName_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LaplaceModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( action == Actions::GET_MATRIX0
    || action == Actions::GET_INT_VECTOR )
  {
    Ref<MatrixBuilder>  mbuilder;
    Vector  disp;
    Vector  intForce;


    // Get the current displacements.
    StateVector::get ( disp, dofs_, globdat );

    // Get the matrix builder and the internal force vector.
    params.find( mbuilder, ActionParams::MATRIX0 );
    params.get ( intForce, ActionParams::INT_VECTOR );

    getMatrix_ ( mbuilder, intForce, disp );

    globdat.set ( ActionParams::MATRIX0, mbuilder );
    globdat.set ( ActionParams::INT_VECTOR, intForce );

    return true;
  }

  if ( action == Actions::GET_MATRIX2 )
  {
    Ref<MatrixBuilder> mbuilder;
    params.get ( mbuilder, ActionParams::MATRIX2 );
    globdat.set ( ActionParams::MATRIX2, mbuilder );

    getMatrix2_( mbuilder );

    return true;
  }

  if ( action == Actions::GET_EXT_VECTOR )
  {
    Vector  extForce;
    params.get ( extForce, ActionParams::EXT_VECTOR );
    globdat.set ( ActionParams::EXT_VECTOR, extForce );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   getMatrix_
//-----------------------------------------------------------------------


void LaplaceModel::getMatrix_

  ( Ref<MatrixBuilder>  mbuilder,
    const Vector&       force,
    const Vector&       disp ) const

{
  Matrix      coords     ( rank_, nodeCount_ );

  Matrix      elemMat    ( nodeCount_, nodeCount_  );
  Vector      elemForce  ( nodeCount_ );
  Vector      elemDisp   ( nodeCount_ );

  Matrix      b          ( rank_, nodeCount_ );
  Matrix      bt         = b.transpose ();

  Cubix       ipGrads    ( rank_, nodeCount_, ipCount_  );
  Vector      ipWeights  ( ipCount_ );
  IdxVector   inodes     ( nodeCount_ );
  IdxVector   idofs      ( nodeCount_  );

  MChain1     mc1;
  MChain2     mc2;
  MChain3     mc3;

  idx_t ipoint = 0;

  // Iterate over all elements assigned to this model.
  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    // Get the global element index.
    idx_t  ielem = ielems_[ie];

    // Get the element coordinates and DOFs.
    elems_.getElemNodes  ( inodes, ielem    );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, dofType_ );

    // Get the gradients and weights
    shape_->getShapeGradients ( ipGrads, ipWeights, coords );

    // Get the displacements at the element nodes.
    elemDisp = select ( disp, idofs );

    // Assemble the element matrix.
    elemMat   = 0.0;
    elemForce = 0.0;

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {
      // Compute the B-matrix for this integration point.
      // Compute the strain vector of this integration point
      getShapeGrads_(b, ipGrads[ip]);

      // Compute the stiffness matrix
      elemMat   += ipWeights[ip] * mc2.matmul ( bt, b );

      ++ipoint;
    }

    // Add the element matrix to the global stiffness matrix.
    if ( mbuilder != NIL )
    {
      mbuilder->addBlock ( idofs, idofs, elemMat );
    }

    // Add the element force vector to the global force vector.
    select ( force, idofs ) += elemForce;
  }
}

//-----------------------------------------------------------------------
//   getMatrix2_
//-----------------------------------------------------------------------

// compute the mass matrix
// current implementation: consistent mass matrix

void LaplaceModel::getMatrix2_

    ( Ref<MatrixBuilder> mbuilder )
{
  Matrix      coords     ( rank_, nodeCount_ );
  Matrix      elemMat    ( nodeCount_, nodeCount_ );

  Matrix      R          ( rank_, rank_ );

  Matrix      sfuncs     = shape_->getShapeFunctions ();
  Matrix      N          ( 1, nodeCount_ );
  Matrix      Nt         = N.transpose ( );

  IdxVector   inodes     ( nodeCount_ );
  IdxVector   idofs      ( nodeCount_  );

  Vector      ipWeights  ( ipCount_   );

  MChain3     mc3;

  double      rho = 1.;

  R = 0.0;

  for ( idx_t i = 0; i < rank_ ; i++ )
  {
    R(i,i) = rho;
  }

  // Iterate over all elements assigned to this model.
  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    // Get the global element index.
    idx_t  ielem = ielems_[ie];

    // Get the element coordinates and DOFs.
    elems_.getElemNodes  ( inodes, ielem    );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, dofType_ );

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


//-----------------------------------------------------------------------
//   getShapeGrads_
//-----------------------------------------------------------------------

// compute the B matrix

void LaplaceModel::getShapeGrads_

  ( const Matrix&   b,
    const Matrix&   g ) const

{
  JEM_ASSERT ( b.size(0) == rank_ &&
                 g.size(0) == rank_ &&
                 b.size(1) == g.size(1) );

  const idx_t  nodeCount = g.size (1);

  b = 0.0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    for (idx_t j = 0; j < rank_; j++ )
    {
      b(j,inode) = g(j,inode);
    }
  }
}


//-----------------------------------------------------------------------
//   getShapeFuncs_
//-----------------------------------------------------------------------

// compute the N matrix

void LaplaceModel::getShapeFuncs_

  ( const Matrix&       s,
    const Vector&       n ) const
{
  JEM_ASSERT ( s.size(0) == 1 &&
               s.size(1) == n.size() );

  const idx_t  nodeCount = n.size ();

  s = 0.0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    s(0,inode) = n[inode];
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newLaplaceModel
//-----------------------------------------------------------------------


static Ref<Model>     newLaplaceModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<LaplaceModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declareLaplaceModel
//-----------------------------------------------------------------------


void declareLaplaceModel ()
{
  using jive::model::ModelFactory;
  ModelFactory::declare ( "Laplace", & newLaplaceModel );
}

#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/array/tensor.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/util/StringUtils.h>

#include <jive/geom/IShapeFactory.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/utilities.h>

#include "MaterialFactory.h"

#include "ElasticModel.h"

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

const char* ElasticModel::DOF_NAMES[3]     = {"dx","dy","dz"};
const char* ElasticModel::SHAPE_PROP       = "shape";
const char* ElasticModel::MATERIAL_PROP    = "material";
const char* ElasticModel::THICK_PROP       = "thickness";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ElasticModel::ElasticModel

   ( const String&       name,
     const Properties&   conf,
     const Properties&   props,
     const Properties&   globdat ) : Super(name)
{

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
  strCount_  = STRAIN_COUNTS[rank_];

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

  // Configure the material
  String matProp = joinNames ( myName_, MATERIAL_PROP );
  material_ = dynamicCast<LinearMaterial> ( 
      MaterialFactory::newInstance ( matProp, conf, props, globdat )
  );

  getShapeGrads_ = getShapeGradsFunc ( rank_ );

  // In 2D, get the thickness (optionally)
  thickness_ = 1.;

  if ( rank_ == 2 )
  {
    myProps.find( thickness_, THICK_PROP );
    myConf.set  ( THICK_PROP, thickness_ );
  }
}

ElasticModel::~ElasticModel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void ElasticModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties  myProps  = props.findProps ( myName_ );
  Properties  matProps = myProps.findProps ( MATERIAL_PROP );
  material_->configure ( matProps, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ElasticModel::getConfig 

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties  myConf  = conf.makeProps ( myName_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ElasticModel::takeAction

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
    Vector  force;

    // Get the current displacements.
    StateVector::get ( disp, dofs_, globdat );

    // Get the matrix builder and the internal force vector.
    params.find( mbuilder, ActionParams::MATRIX0 );
    params.get ( force,    ActionParams::INT_VECTOR );

    getMatrix_ ( mbuilder, force, disp );
    return true;
  }

  if ( action == Actions::GET_MATRIX2 )
  {
    Ref<MatrixBuilder> mbuilder;
    params.get ( mbuilder, ActionParams::MATRIX2 );

    getMatrix2_( *mbuilder );
    return true;
  }

  /*
  if ( action == Actions::GET_TABLE )
  {
    return getTable_ ( params, globdat );
  }
  */

  return false;
}


//-----------------------------------------------------------------------
//   getMatrix_
//-----------------------------------------------------------------------


void ElasticModel::getMatrix_

  ( Ref<MatrixBuilder>  mbuilder,
    const Vector&       force,
    const Vector&       disp ) const

{
  Matrix      stiff      ( strCount_, strCount_ );
  Matrix      coords     ( rank_, nodeCount_ );

  Matrix      elemMat    ( dofCount_, dofCount_  );
  Vector      elemForce  ( dofCount_ );
  Vector      elemDisp   ( dofCount_ );

  Vector      strain     ( strCount_ );
  Vector      stress     ( strCount_ );

  Matrix      b          ( strCount_, dofCount_  );
  Matrix      bt         = b.transpose ();

  Cubix       ipGrads    ( rank_, nodeCount_, ipCount_  );
  Vector      ipWeights  ( ipCount_ );
  IdxVector   inodes     ( nodeCount_ );
  IdxVector   idofs      ( dofCount_  );

  MChain1     mc1;
  MChain3     mc3;

  // Iterate over all elements assigned to this model.
  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    // Get the global element index.
    idx_t  ielem = ielems_[ie];

    // Get the element coordinates and DOFs.
    elems_.getElemNodes  ( inodes, ielem    );
    nodes_.getSomeCoords ( coords, inodes );
    dofs_->getDofIndices ( idofs,  inodes, dofTypes_ );

    // Get the gradients and weights
    shape_->getShapeGradients ( ipGrads, ipWeights, coords );

    // for 2D: multiply ipWeights with thickness
    ipWeights *= thickness_;

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
      material_->stiffAtPoint ( stiff, ip );
      elemMat   += ipWeights[ip] * mc3.matmul ( bt, stiff, b );
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

void ElasticModel::getMatrix2_

    ( MatrixBuilder&          mbuilder )
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

  double      rho = 0.0024;

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
    mbuilder.addBlock ( idofs, idofs, elemMat );
  }
}


/*
//-----------------------------------------------------------------------
//   getTable_
//-----------------------------------------------------------------------


bool ElasticModel::getTable_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;
  using jive::model::StateVector;

  String       contents;
  Ref<XTable>  table;
  Vector       weights;
  String       name;

  Vector       disp;

  StateVector::get ( disp, dofs_, globdat );

  // Get the table, the name of the table, and the table row weights
  // from the action parameters.

  params.get ( table,   ActionParams::TABLE );
  params.get ( name,    ActionParams::TABLE_NAME );
  params.get ( weights, ActionParams::TABLE_WEIGHTS );

  // Stress value are computed in the nodes.

  if ( name == "stress" &&
       table->getRowItems() == nodes_.getData() )
  {
    getStress_ ( *table, weights, disp );

    return true;
  }
  else if ( name == "xoutTable" )
  {
    params.get ( contents, "contentString" );

    getXOutTable_ ( table, weights, contents, disp );

    return true;
  }
  return false;
}

//-----------------------------------------------------------------------
//   getStress_
//-----------------------------------------------------------------------


void ElasticModel::getStress_

  ( XTable&        table,
    const Vector&  weights,
    const Vector&  disp )

{
  IdxVector   ielems     = egroup_.getIndices  ();

  Matrix     ndNStress  ( nodeCount_, strCount_ );  // nodal normal stress
  Vector     ndWeights  ( nodeCount_ );

  Matrix     coords     ( rank_,     nodeCount_ );
  Matrix     b          ( strCount_, dofCount_  );

  Vector     nStressIp  ( strCount_ );    // normal stress vector at idx_t.pt.
  Vector     strain     ( strCount_ );
  Vector     elemDisp   ( dofCount_ );

  IdxVector  inodes     ( nodeCount_ );
  IdxVector  idofs      ( dofCount_  );
  IdxVector  jcols      ( strCount_  );

  jcols.resize ( strCount_ );

  // Add the columns for the stress components to the table.

  switch ( strCount_ )
  {
  case 1:

    jcols[0] = table.addColumn ( "xx" );

    break;

  case 3:

    jcols[0] = table.addColumn ( "xx" );
    jcols[1] = table.addColumn ( "yy" );
    jcols[2] = table.addColumn ( "xy" );

    break;

  case 6:

    jcols[0] = table.addColumn ( "xx" );
    jcols[1] = table.addColumn ( "yy" );
    jcols[2] = table.addColumn ( "zz" );
    jcols[3] = table.addColumn ( "xy" );
    jcols[4] = table.addColumn ( "yz" );
    jcols[5] = table.addColumn ( "xz" );

    break;

  default:

    throw Error (
      JEM_FUNC,
      "unexpected number of stress components: " +
      String ( strCount_ )
    );
  }

  idx_t         ipoint = 0;

  Vector      ipWeights ( ipCount_ );

  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    // Get the global element index.

    idx_t  ielem = ielems[ie];

    ndNStress  = 0.0;
    ndWeights  = 0.0;

    elems_.getElemNodes  ( inodes, ielem );
    dofs_->getDofIndices ( idofs,  inodes,  dofTypes_ );

    nodes_.getSomeCoords ( coords, inodes );

    shape_->setGradsForIntegration ( ipWeights, coords, ie );

    elemDisp = select ( disp, idofs );

    Matrix     sfuncs     = shape_->getShapeFunctions ();

    // Iterate over the integration points.

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {
      shape_->getStrain ( strain, b, elemDisp, ip, ie );

      material_->stressAtPoint ( nStressIp, strain, ipoint );

      ndNStress += matmul ( sfuncs(ALL,ip), nStressIp );

      ndWeights += sfuncs(ALL,ip);

      ++ipoint; 
    }

    select ( weights, inodes ) += ndWeights;

    // Add the stresses to the table.

    table.addBlock ( inodes, jcols[slice(0,strCount_)],   ndNStress );
  }
}
*/

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newElasticModel
//-----------------------------------------------------------------------


static Ref<Model>     newElasticModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<ElasticModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declareElasticModel
//-----------------------------------------------------------------------


void declareElasticModel ()
{
  using jive::model::ModelFactory;
  ModelFactory::declare ( "Elastic", & newElasticModel );
}


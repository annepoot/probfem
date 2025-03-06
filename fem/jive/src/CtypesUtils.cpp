#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/geom/Names.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/util/XDofSpace.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/util/Constraints.h>
#include <jive/util/utilities.h>

#include "CtypesUtils.h"

using namespace jem;

using jive::Vector;
using jive::IdxVector;
using jive::algebra::AbstractMatrix;
using jive::algebra::SparseMatrixObject;
using jive::algebra::MatrixBuilder;
using jive::geom::PropertyNames;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::model::ActionParams;
using jive::model::STATE0;
using jive::model::StateVector;
using jive::util::Constraints;
using jive::util::XDofSpace;
using jive::util::joinNames;


#ifdef __cplusplus
extern "C" {
#endif

long NODESET_MASK       = 1 << 0;
long ELEMENTSET_MASK    = 1 << 1;
long DOFSPACE_MASK      = 1 << 2;
long STATE0_MASK        = 1 << 3;
long EXTFORCE_MASK      = 1 << 4;
long INTFORCE_MASK      = 1 << 5;
long MATRIX0_MASK       = 1 << 6;
long CONSTRAINTS_MASK   = 1 << 7;
long SHAPE_MASK         = 1 << 8;


void globdatToCtypes
	( GLOBDAT& iodat,
	  Properties globdat,
	  long flags )
{
  ElementSet elems = ElementSet::get( globdat, "" );
  idx_t elemCount = elems.size();
  idx_t maxElemNodeCount = elems.maxElemNodeCount();

  NodeSet nodes = elems.getNodes();
  idx_t nodeCount = nodes.size();
  idx_t rank = nodes.rank();

  Ref<XDofSpace> dofs = XDofSpace::get ( nodes.getData(), globdat );
  idx_t typeCount = dofs->typeCount();
  idx_t dofCount = dofs->dofCount();

  if ( (flags & NODESET_MASK) > 0 ){

    // Populate nodeSet array
    if ( nodeCount > iodat.nodeSet.data.shape[0] || rank > iodat.nodeSet.data.shape[1] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
    
    Vector coords (rank);
    for ( idx_t inode = 0; inode < nodeCount ; inode++ ){
      nodes.getNodeCoords(coords, inode);
    
      for ( idx_t ir = 0; ir < rank ; ir++ ){
        iodat.nodeSet.data.ptr[(inode * rank) + ir] = coords[ir];
      }
    }
    iodat.nodeSet.data.shape[0] = nodeCount;
    iodat.nodeSet.data.shape[1] = rank;
  }

  if ( (flags & ELEMENTSET_MASK) > 0 ){
    // Populate elementSet array
    if ( elemCount > iodat.elementSet.data.shape[0] ||
         maxElemNodeCount > iodat.elementSet.data.shape[1] ||
         elemCount > iodat.elementSet.sizes.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    IdxVector inodes (maxElemNodeCount);
    for ( idx_t ielem = 0; ielem < elemCount; ielem++ ){
      idx_t elemNodeCount = elems.getElemNodeCount(ielem);
      elems.getElemNodes(inodes, ielem);

      iodat.elementSet.sizes.ptr[ielem] = elemNodeCount;
      for ( idx_t in = 0; in < elemNodeCount; in++ ){
        iodat.elementSet.data.ptr[(ielem * maxElemNodeCount) + in] = inodes[in];
      }
    }
    iodat.elementSet.data.shape[0] = elemCount;
    iodat.elementSet.data.shape[1] = maxElemNodeCount;
    iodat.elementSet.sizes.shape[0] = elemCount;
  }

  if ( (flags & DOFSPACE_MASK) > 0 ){
    // Populate dofSpace array
    if ( nodeCount > iodat.dofSpace.data.shape[0] || typeCount > iodat.dofSpace.data.shape[1] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    for ( int inode = 0; inode < nodeCount; inode++ ){
      for ( int itype = 0; itype < typeCount; itype++ ){
  	    iodat.dofSpace.data.ptr[(inode * typeCount) + itype] = dofs->getDofIndex(inode, itype);
  	  }
    }

    iodat.dofSpace.data.shape[0] = nodeCount;
    iodat.dofSpace.data.shape[1] = typeCount;
  }

  if ( (flags & STATE0_MASK) > 0 ){
    Vector u ( dofs->dofCount() );
    StateVector::get ( u, STATE0, dofs, globdat );

    // Populate state0 array
    if ( dofCount > iodat.state0.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    for ( idx_t idof = 0; idof < dofCount ; idof++ ){
      iodat.state0.ptr[idof] = u[idof];
    }
    iodat.state0.shape[0] = dofCount;
  }

  if ( (flags & EXTFORCE_MASK) > 0 ){
    Vector fext;
    globdat.get( fext, ActionParams::EXT_VECTOR );

    // Populate extForce array
    if ( dofCount > iodat.extForce.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    for ( idx_t idof = 0; idof < dofCount ; idof++ ){
      iodat.extForce.ptr[idof] = fext[idof];
    }
    iodat.extForce.shape[0] = dofCount;
  }

  if ( (flags & INTFORCE_MASK) > 0 ){
    Vector fint;
    globdat.get( fint, ActionParams::INT_VECTOR );
  
    // Populate intForce array
    if ( dofCount > iodat.intForce.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    for ( idx_t idof = 0; idof < dofCount ; idof++ ){
      iodat.intForce.ptr[idof] = fint[idof];
    }
    iodat.intForce.shape[0] = dofCount;
  }

  if ( (flags & MATRIX0_MASK) > 0 ){
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

    if ( indexCount > iodat.matrix0.values.shape[0] ||
         indexCount > iodat.matrix0.indices.shape[0] ||
         offsetCount > iodat.matrix0.offsets.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
  
    for ( idx_t i = 0; i < indexCount ; i++ ){
      iodat.matrix0.values.ptr[i] = values[i];
      iodat.matrix0.indices.ptr[i] = colIndices[i];
    }
  
    for ( idx_t i = 0; i < offsetCount ; i++ ){
      iodat.matrix0.offsets.ptr[i] = rowOffsets[i];
    }
  
    iodat.matrix0.values.shape[0] = indexCount;
    iodat.matrix0.indices.shape[0] = indexCount;
    iodat.matrix0.offsets.shape[0] = offsetCount;
  }

  if ( (flags & CONSTRAINTS_MASK) > 0 ){
    Ref<Constraints> cons = Constraints::get ( dofs, globdat );
    idx_t cdofCount = cons->slaveDofCount();

    // Populate constraints array
    if ( cons->masterDofCount() > 0 ){
      throw Exception ( "getState0()", "master dofs have not been implemented");
    }
    
    if ( cdofCount > iodat.constraints.dofs.shape[0] ){
      throw Exception ( "getState0()", "buffer size insufficient");
    }
    
    IdxVector cdofs = cons->getSlaveDofs();
    Vector cvals ( cdofs.size() );
    cons->getRvalues(cvals, cdofs);
    
    for ( idx_t idof = 0; idof < cdofCount ; idof++ ){
      iodat.constraints.dofs.ptr[idof] = cdofs[idof];
      iodat.constraints.values.ptr[idof] = cvals[idof];
    }
    iodat.constraints.dofs.shape[0] = cdofCount;
    iodat.constraints.values.shape[0] = cdofCount;
  }

  if ( (flags & SHAPE_MASK) > 0 ){
    // Populate shape string
    String shapeType;
    String shapeScheme;
  
    globdat.get(shapeType, joinNames(PropertyNames::SHAPE, PropertyNames::TYPE));
    globdat.get(shapeScheme, joinNames(PropertyNames::SHAPE, PropertyNames::ISCHEME));
  
    int shapeTypeSize = shapeType.size();
    int shapeSchemeSize = shapeScheme.size();
  
    for ( idx_t is = 0; is < shapeTypeSize; is++ ){
      iodat.shape.type[is] = shapeType[is];
    }
    for ( idx_t is = 0; is < shapeSchemeSize; is++ ){
      iodat.shape.ischeme[is] = shapeScheme[is];
    }
  }
}

#ifdef __cplusplus
}
#endif

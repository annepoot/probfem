#ifndef ELASTIC_MODEL_H
#define ELASTIC_MODEL_H

#include <jem/util/Properties.h>

#include <jive/algebra/MatrixBuilder.h>
#include <jive/fem/ElementGroup.h>
#include <jive/geom/InternalShape.h>
#include <jive/model/Model.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

#include "LinearMaterial.h"
#include "utilities.h"

using namespace jem;

using jem::util::Properties;
using jive::Vector;
using jive::IdxVector;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::algebra::MatrixBuilder;
using jive::model::Model;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::ElementGroup;
using jive::geom::InternalShape;

typedef ElementSet              ElemSet;
typedef ElementGroup            ElemGroup;

class ElasticModel : public Model
{
 public:

  typedef ElasticModel         Self;
  typedef Model              Super;

  static const char*         DOF_NAMES[3];
  static const char*         SHAPE_PROP;
  static const char*         MATERIAL_PROP;
  static const char*         THICK_PROP;
  static const char*	       LARGE_DISP_PROP;

                       ElasticModel
			 
    ( const String&       name,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );

  virtual void         configure

    ( const Properties&   props,
      const Properties&   globdat );

  virtual void         getConfig

    ( const Properties&   conf,
      const Properties&   globdat )      const;

  virtual bool         takeAction

    ( const String&       action,
      const Properties&   params,
      const Properties&   globdat );

 protected:

  virtual              ~ElasticModel ();

  virtual void         getMatrix_

    ( Ref<MatrixBuilder>  mbuilder,
      const Vector&       force,
      const Vector&       disp )       const;

  void                 getMatrix2_

    ( MatrixBuilder&      mbuilder );

/*
  bool                 getTable_

    ( const Properties&   params,
      const Properties&   globdat );
 
  void                 getStress_

    ( XTable&             table,
      const Vector&       weights,
      const Vector&       disp );

 void                  getStrain_

    ( 	    Vector&           strain,
            Matrix&           b,
      const Vector&           disp )    const;
*/

 protected:

  Assignable<ElemGroup>   egroup_;
  Assignable<ElemSet>     elems_;
  Assignable<NodeSet>     nodes_;

  IdxVector               ielems_;

  idx_t                   rank_;
  idx_t                   nodeCount_;
  idx_t                   numElem_;
  idx_t                   numNode_;
  idx_t                   strCount_;
  idx_t                   dofCount_;
  idx_t                   ipCount_;
  double                  thickness_;

  Ref<InternalShape>      shape_;
  Ref<LinearMaterial>     material_;

  Ref<XDofSpace>          dofs_;
  IdxVector               dofTypes_;

  ShapeGradsFunc          getShapeGrads_;
  ShapeFuncsFunc          getShapeFuncs_;

  String                  myTag_;
};

#endif

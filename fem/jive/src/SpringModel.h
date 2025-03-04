#ifndef SPRING_MODEL_H
#define SPRING_MODEL_H

#include <jem/util/Properties.h>

#include <jive/algebra/MatrixBuilder.h>
#include <jive/fem/ElementGroup.h>
#include <jive/geom/InternalShape.h>
#include <jive/model/Model.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

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

class SpringModel : public Model
{
 public:

  typedef SpringModel         Self;
  typedef Model              Super;

  static const char*         DOF_NAMES[3];
  static const char*         K_PROP;
  static const char*         SHAPE_PROP;

                       SpringModel
			 
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

  virtual              ~SpringModel ();

  virtual void         getMatrix_

    ( Ref<MatrixBuilder>  mbuilder )       const;

 protected:

  Assignable<ElemGroup>   egroup_;
  Assignable<ElemSet>     elems_;
  Assignable<NodeSet>     nodes_;

  IdxVector               ielems_;

  idx_t                   rank_;
  idx_t                   nodeCount_;
  idx_t                   numElem_;
  idx_t                   numNode_;
  idx_t                   dofCount_;
  idx_t                   ipCount_;

  double                  k_;

  Ref<InternalShape>      shape_;

  Ref<XDofSpace>          dofs_;
  IdxVector               dofTypes_;

  ShapeFuncsFunc          getShapeFuncs_;

  String                  myTag_;
};

#endif

#ifndef LAPLACE_MODEL_H
#define LAPLACE_MODEL_H

#include <jem/util/Properties.h>

#include <jive/algebra/MatrixBuilder.h>
#include <jive/fem/ElementGroup.h>
#include <jive/geom/InternalShape.h>
#include <jive/model/Model.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

using namespace jem;

using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
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

class LaplaceModel : public Model
{
 public:

  typedef LaplaceModel         Self;
  typedef Model              Super;

  static const char*         DOF_NAMES[3];
  static const char*         SHAPE_PROP;

                       LaplaceModel

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

  virtual              ~LaplaceModel ();

  virtual void         getMatrix_

    ( Ref<MatrixBuilder>  mbuilder,
      const Vector&       force,
      const Vector&       disp )       const;

  void                 getMatrix2_

    ( Ref<MatrixBuilder> mbuilder );

  void                  getShapeGrads_

    ( const Matrix&       b,
      const Matrix&       g )          const;

  void                  getShapeFuncs_

    ( const Matrix&       sfuncs,
      const Vector&       n )          const;

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

  Ref<InternalShape>      shape_;

  Ref<XDofSpace>          dofs_;
  idx_t                   dofType_;

  String                  myTag_;
};

#endif

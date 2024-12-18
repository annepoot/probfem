/*
 *  TU Delft / Knowledge Centre WMC
 *
 *  Iuri Barcelos, August 2015
 *
 *  Simple base class for materials.
 *
 */

#ifndef LINEAR_ISOTROPIC_MATERIAL_H
#define LINEAR_ISOTROPIC_MATERIAL_H

#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/XTable.h>

#include "Material.h"
#include "LinearMaterial.h"
#include "IsotropicMaterial.h"

using jem::System;
using jem::idx_t;
using jem::Ref;
using jem::Object;
using jem::String;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::BoolVector;
using jive::StringVector;
using jive::util::XTable;


//-----------------------------------------------------------------------
//   class LinearIsotropicMaterial
//-----------------------------------------------------------------------

class LinearIsotropicMaterial : public LinearMaterial, public IsotropicMaterial
{
 public:

  typedef LinearIsotropicMaterial Self;

   explicit              LinearIsotropicMaterial

    ( const String&        name,
      const Properties&    props,
      const Properties&    conf,
      const Properties&    globdat );

  virtual void           configure
  
    ( const Properties&    props,
      const Properties&    globdat );

  virtual void           getConfig

    ( const Properties&    conf,
      const Properties&    globdat ) const;

  virtual void           stiffAtPoint

    ( Matrix&              stiff,
      const idx_t          ipoint );

  virtual void           stressAtPoint

    ( Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  static Ref<Material>      makeNew

    ( const String&         mat,
      const Properties&     conf,
      const Properties&     props,
      const Properties&     globdat );

  static void             declare          ();

 protected:
                       ~LinearIsotropicMaterial ();

    Matrix stiff_;
};

#endif

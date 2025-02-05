/*
 *  TU Delft / Knowledge Centre WMC
 *
 *  Iuri Barcelos, August 2015
 *
 *  Simple base class for materials.
 *
 */

#include <jem/base/Error.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>

#include "MaterialFactory.h"
#include "LinearIsotropicMaterial.h"

using namespace jem;
using jem::numeric::matmul;

//=======================================================================
//   class LinearIsotropicMaterial
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LinearIsotropicMaterial::LinearIsotropicMaterial
  
  ( const String&        name,
    const Properties&    props,
    const Properties&    conf,
    const Properties&    globdat ):

  Material ( name, props, conf, globdat ),
  LinearMaterial ( name, props, conf, globdat ),
  IsotropicMaterial ( name, props, conf, globdat )

{}


LinearIsotropicMaterial::~LinearIsotropicMaterial ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void LinearIsotropicMaterial::configure

  ( const Properties& props,
    const Properties& globdat )
{
  LinearMaterial::configure ( props, globdat );
  IsotropicMaterial::configure ( props, globdat );
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void LinearIsotropicMaterial::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{
  LinearMaterial::getConfig ( props, globdat );
  IsotropicMaterial::getConfig ( props, globdat );
}

//-----------------------------------------------------------------------
//   stiffAtPoint
//-----------------------------------------------------------------------

void LinearIsotropicMaterial::stiffAtPoint

  ( Matrix&              stiff,
    const idx_t          ipoint )

{
  IsotropicMaterial::stiffAtPoint(stiff, ipoint);
}

//-----------------------------------------------------------------------
//   stressAtPoint
//-----------------------------------------------------------------------

void LinearIsotropicMaterial::stressAtPoint

  ( Vector&              stress,
    const Vector&        strain,
    const idx_t          ipoint )

{
  IsotropicMaterial::stressAtPoint(stress, strain, ipoint);
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Material> LinearIsotropicMaterial::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------

void LinearIsotropicMaterial::declare ()
{
  MaterialFactory::declare ( "LinearIsotropic", & makeNew );
}

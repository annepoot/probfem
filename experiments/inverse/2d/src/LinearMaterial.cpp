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

#include "LinearMaterial.h"

using namespace jem;

//=======================================================================
//   class LinearMaterial
//=======================================================================

//-----------------------------------------------------------------------
//   constructors and destructor
//-----------------------------------------------------------------------


LinearMaterial::LinearMaterial

  ( const String&        name,
    const Properties&    props,
    const Properties&    conf,
    const Properties&    globdat ) : 

  Material ( name, props, conf, globdat )

{}


LinearMaterial::~LinearMaterial()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void LinearMaterial::configure

  ( const Properties& props,
    const Properties& globdat )
{}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void LinearMaterial::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{}


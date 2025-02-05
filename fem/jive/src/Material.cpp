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

#include "Material.h"
#include "IsotropicMaterial.h"

using namespace jem;


//=======================================================================
//   class Material
//=======================================================================

//-----------------------------------------------------------------------
//   constructors and destructor
//-----------------------------------------------------------------------

Material::Material

  ( const String&        name,
    const Properties&    props,
    const Properties&    conf,
    const Properties&    globdat )

{}

Material::~Material()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void Material::configure

  ( const Properties& props,
    const Properties& globdat )
{}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void Material::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void Material::update

  ( Matrix&              stiff,
    Vector&              stress,
    const Vector&        strain,
    const idx_t          ipoint )

{}

//-----------------------------------------------------------------------
//   stressAtPoint
//-----------------------------------------------------------------------

void Material::stressAtPoint

  ( Vector&              stress,
    const Vector&        strain,
    const idx_t          ipoint )

{}

//-----------------------------------------------------------------------
//   addTableColumns
//-----------------------------------------------------------------------

void Material::addTableColumns

  ( IdxVector&           jcols,
    XTable&              table,
    const String&        name )

{}

//-----------------------------------------------------------------------
//   hasThermal
//-----------------------------------------------------------------------

bool Material::hasThermal ()
{
  // Default implementation.

  return false;
}

//-----------------------------------------------------------------------
//   hasSwelling
//-----------------------------------------------------------------------

bool Material::hasSwelling ()
{
  // Default implementation.

  return false;
}

//-----------------------------------------------------------------------
//   hasCrackBand
//-----------------------------------------------------------------------

bool Material::hasCrackBand ()
{
  // Default implementation.

  return false;
}

//-----------------------------------------------------------------------
//   setCharLength
//-----------------------------------------------------------------------

void Material::setCharLength

  ( const idx_t       ipoint,
    const double      le )

{
  // Default implementation
}


//-----------------------------------------------------------------------
//   getHistoryCount
//-----------------------------------------------------------------------

idx_t Material::getHistoryCount () const
{
  return historyNames_.size();
}

//-----------------------------------------------------------------------
//   getHistoryNames
//-----------------------------------------------------------------------

StringVector Material::getHistoryNames

  () const
{
  return historyNames_;
}

//-----------------------------------------------------------------------
//   getHistoryNames
//-----------------------------------------------------------------------

void Material::getHistoryNames

  ( const StringVector&  hnames ) const
{
  JEM_ASSERT ( hnames.size() == getHistoryCount() );
  hnames = historyNames_;
}

//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

void Material::getHistory

  ( Vector&        history,
    const idx_t    ipoint )

{
}

//-----------------------------------------------------------------------
//   setHistory
//-----------------------------------------------------------------------

void Material::setHistory

  ( const Vector&    hvals,
    const idx_t      mpoint )

{
  // Default implementation
}

//-----------------------------------------------------------------------
//   copyHistory
//-----------------------------------------------------------------------

void Material::copyHistory

  ( const idx_t      master,
    const idx_t      slave )

{
  // Default implementation
}

//-----------------------------------------------------------------------
//   createIntPoints
//-----------------------------------------------------------------------

void Material::createIntPoints

  ( const idx_t       npoints )

{}

void Material::createIntPoints

  ( const Matrix& ipCoords )

{}

//-----------------------------------------------------------------------
//   setConc
//-----------------------------------------------------------------------

void Material::setConc

  ( const idx_t       ipoint,
    const double      conc )

{
}

//-----------------------------------------------------------------------
//   setDeltaT
//-----------------------------------------------------------------------

void Material::setDeltaT

  ( const idx_t       ipoint,
    const double      deltaT )

{
}

//-----------------------------------------------------------------------
//   commit 
//-----------------------------------------------------------------------

void Material::commit ()
{}

//-----------------------------------------------------------------------
//   checkCommit 
//-----------------------------------------------------------------------

void Material::checkCommit 

  ( const Properties& params )

{}

//-----------------------------------------------------------------------
//   commitOne
//-----------------------------------------------------------------------

void Material::commitOne 

 ( const idx_t ipoint )

{}

//-----------------------------------------------------------------------
//   cancel 
//-----------------------------------------------------------------------

void Material::cancel ()
{}

//-----------------------------------------------------------------------
//   getFileName 
//-----------------------------------------------------------------------

String Material::getFileName

  ( const idx_t      ipoint ) const
{
  return "";
}

//-----------------------------------------------------------------------
//   getDissipation 
//-----------------------------------------------------------------------

double Material::getDissipation

  ( const idx_t      ipoint ) const
{
  return 0.;
}

//-----------------------------------------------------------------------
//   getDissipationGrad
//-----------------------------------------------------------------------

double Material::getDissipationGrad

  ( const idx_t      ipoint ) const
{
  return 0.;
}

//-----------------------------------------------------------------------
//   pointCount
//-----------------------------------------------------------------------

idx_t Material::pointCount () const
{
  return 0;
}

//-----------------------------------------------------------------------
//   isLoading 
//-----------------------------------------------------------------------

bool  Material::isLoading 

  ( idx_t ipoint ) const

{
  return false;
}

//-----------------------------------------------------------------------
//   wasLoading 
//-----------------------------------------------------------------------

bool  Material::wasLoading 

  ( idx_t ipoint ) const

{
  return false;
}

//-----------------------------------------------------------------------
//   isInelastic
//-----------------------------------------------------------------------

bool Material::isInelastic

  ( idx_t ipoint ) const

{
  return false;
}

//-----------------------------------------------------------------------
//   despair 
//-----------------------------------------------------------------------

bool Material::despair () 

{
  desperateMode_ = true;

  idx_t np = pointCount();

  hasSwitched_.resize ( np );
  useSecant_  .resize ( np );

  useSecant_ = false;

  for ( idx_t ip = 0; ip < np; ++ip )
  {
    hasSwitched_[ip] = ( isLoading(ip) != wasLoading(ip) );
  }

  return np > 0;
}

//-----------------------------------------------------------------------
//   endDespair 
//-----------------------------------------------------------------------

void Material::endDespair ()

{
  desperateMode_ = false;
  useSecant_ = false;
}

//-----------------------------------------------------------------------
//   writeState 
//-----------------------------------------------------------------------

void Material::writeState ()

{
}


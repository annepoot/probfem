/*
 *  TU Delft / Knowledge Centre WMC
 *
 *  Iuri Barcelos, August 2015
 *
 *  Simple isotropic material class to compute stiffness and stresses
 *  as well as store relevant data.
 *
 */

#include <jem/io/PrintWriter.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>

#include "IsotropicMaterial.h"

#include "utilities.h"

#include <cstdlib>

using namespace jem;
using jem::numeric::matmul;
using jem::numeric::norm2;
using jem::io::endl;

using jem::io::PrintWriter;

//=======================================================================
//   class IsotropicMaterial
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* IsotropicMaterial::E_PROP        = "E";
const char* IsotropicMaterial::NU_PROP       = "nu";
const char* IsotropicMaterial::RANK_PROP     = "rank";
const char* IsotropicMaterial::AREA_PROP     = "area";
const char* IsotropicMaterial::ANMODEL_PROP  = "anmodel";
const char* IsotropicMaterial::SWELLING_PROP = "swelling_coeff";
const char* IsotropicMaterial::ALPHA_PROP    = "alpha";

//-----------------------------------------------------------------------
//   constructor and destructor
//-----------------------------------------------------------------------

IsotropicMaterial::IsotropicMaterial

  ( const String&        name,
    const Properties&    props,
    const Properties&    conf,
    const Properties&    globdat ) : 

  Material ( name, props, conf, globdat )

{
  e_       = 1.0;
  nu_      = 0.0;
  area_    = 1.0;
  alpha_   = 0.0;
  swcoeff_ = 0.0;
}

IsotropicMaterial::~IsotropicMaterial ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void IsotropicMaterial::configure

  ( const Properties& props,
    const Properties& globdat )

{
  props.get  ( e_, E_PROP );
  props.get  ( nu_, NU_PROP );
  props.get  ( rank_, RANK_PROP );
  props.get  ( anmodel_, ANMODEL_PROP );
  props.find ( alpha_, ALPHA_PROP );
  props.find ( swcoeff_, SWELLING_PROP );

  JEM_PRECHECK ( rank_ >= 1 && rank_ <= 3 );

  if ( rank_ == 1 )
    props.get ( area_, AREA_PROP );

  const idx_t STRAIN_COUNTS[4] = { 0, 1, 3, 6 };

  stiffMatrix_.resize ( STRAIN_COUNTS[rank_], STRAIN_COUNTS[rank_] );
  stiffMatrix_ = 0.0;

  computeStiffMatrix_ ();
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void IsotropicMaterial::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const
{
  conf.set   ( E_PROP, e_ );
  conf.set   ( NU_PROP, nu_ );
  conf.set   ( ANMODEL_PROP, anmodel_ );
  conf.set   ( ALPHA_PROP, alpha_ );
  conf.set   ( SWELLING_PROP, swcoeff_ );

  if ( rank_ == 1 )
    conf.set ( AREA_PROP, area_ );
}

//-----------------------------------------------------------------------
//  hasThermal 
//-----------------------------------------------------------------------

bool IsotropicMaterial::hasThermal ()
{
  if ( alpha_ != 0.0 )
    return true;
  else
    return false;
}

//-----------------------------------------------------------------------
//  hasSwelling 
//-----------------------------------------------------------------------

bool IsotropicMaterial::hasSwelling ()
{
  if ( swcoeff_ != 0.0 )
    return true;
  else
    return false;
}

//-----------------------------------------------------------------------
//  createIntPoints 
//-----------------------------------------------------------------------

void IsotropicMaterial::createIntPoints

  ( const idx_t       npoints )

{
  iPointConc_.resize   ( npoints );
  iPointDeltaT_.resize ( npoints );

  iPointConc_   = 0.0;
  iPointDeltaT_ = 0.0;
}

//-----------------------------------------------------------------------
//  setConc 
//-----------------------------------------------------------------------

void IsotropicMaterial::setConc

  ( const idx_t       ipoint,
    const double      conc )

{
  iPointConc_[ipoint] = conc;
}

//-----------------------------------------------------------------------
//  setTemp 
//-----------------------------------------------------------------------

void IsotropicMaterial::setDeltaT

  ( const idx_t       ipoint,
    const double      deltaT )

{
  iPointDeltaT_[ipoint] = deltaT;
}

//-----------------------------------------------------------------------
//  update 
//-----------------------------------------------------------------------

void IsotropicMaterial::update

  ( Matrix&       stiff,
    Vector&       stress,
    const Vector& strain,
    const idx_t   ipoint )
{
  stiff = stiffMatrix_;

  Vector mechStrain ( strain.size() );
  mechStrain = 0.;
  mechStrain = strain;

  if ( alpha_ != 0.0 )
  {
    Vector thermStrain;
    computeThermalStrains_ ( thermStrain, iPointDeltaT_[ipoint] );
    mechStrain -= thermStrain;
  }

  if ( swcoeff_ != 0.0 )
  {
    Vector swellingStrain;
    computeSwellingStrains_ ( swellingStrain, iPointConc_[ipoint] );
    mechStrain -= swellingStrain;
  }

  matmul ( stress, stiffMatrix_, mechStrain );
}

//-----------------------------------------------------------------------
//  stressAtPoint 
//-----------------------------------------------------------------------

void IsotropicMaterial::stressAtPoint

  ( Vector&       stress,
    const Vector& strain,
    const idx_t   ipoint )
{
  const idx_t STRAIN_COUNTS[4] = { 0, 1, 3, 6 };

  Vector strvec ( STRAIN_COUNTS[rank_] );

  Matrix stiff ( stiffMatrix_ );

  update ( stiff, strvec, strain, ipoint );

  if ( anmodel_ == "PLANE_STRAIN" )
  {
    stress[0] = strvec[0];
    stress[1] = strvec[1];
    stress[2] = strvec[2];
    stress[3] = nu_ * ( strvec[0] + strvec[1] );
  }
  else
    stress = strvec;
}

//-----------------------------------------------------------------------
//  stiffAtPoint 
//-----------------------------------------------------------------------

void IsotropicMaterial::stiffAtPoint

  ( Vector&       stiffvec,
    const idx_t   ipoint )
{
  System::warn() << "IsotropicMaterial: stiffAtPoint not implemented.\n";
}

//-----------------------------------------------------------------------
//  strengthAtPoint
//-----------------------------------------------------------------------

double IsotropicMaterial::strengthAtPoint

  ( const idx_t   ipoint )
{
  System::warn() << "IsotropicMaterial: strengthAtPoint not implemented.\n";
  return 0.;
}

//-----------------------------------------------------------------------
//  addTableColumns 
//-----------------------------------------------------------------------

void IsotropicMaterial::addTableColumns

  ( IdxVector&     jcols,
    XTable&        table,
    const String&  name )

{
  // Check if the requested table is supported by this material.

  if ( name == "nodalStress" || name == "ipStress" )
  {
    if ( anmodel_ == "BAR" )
    {
      jcols.resize ( 1 );

      jcols[0] = table.addColumn ( "s_xx" );
    }

    else if ( anmodel_ == "PLANE_STRESS" )
    {
      jcols.resize ( 3 );

      jcols[0] = table.addColumn ( "s_xx" );
      jcols[1] = table.addColumn ( "s_yy" );
      jcols[2] = table.addColumn ( "s_xy" );
    }

    else if ( anmodel_ == "PLANE_STRAIN" )
    {
      jcols.resize ( 4 );

      jcols[0] = table.addColumn ( "s_xx" );
      jcols[1] = table.addColumn ( "s_yy" );
      jcols[2] = table.addColumn ( "s_xy" );
      jcols[3] = table.addColumn ( "s_zz" );
    }

    else if ( anmodel_ == "SOLID" )
    {
      jcols.resize ( 6 );

      jcols[0] = table.addColumn ( "s_xx" );
      jcols[1] = table.addColumn ( "s_yy" );
      jcols[2] = table.addColumn ( "s_zz" );
      jcols[3] = table.addColumn ( "s_xy" );
      jcols[5] = table.addColumn ( "s_yz" );
      jcols[4] = table.addColumn ( "s_zx" );
    }

    else
      throw Error ( JEM_FUNC, "Unexpected analysis model: " + anmodel_ );
  }
}


//-----------------------------------------------------------------------
//  computeStiffMatrix_ 
//-----------------------------------------------------------------------

void IsotropicMaterial::computeStiffMatrix_

  ( )
{

  if ( anmodel_ == "BAR" )
  {
    stiffMatrix_(0,0) = e_ * area_;
  }
  
  else if ( anmodel_ == "PLANE_STRESS" )
  {
    stiffMatrix_(0,0) = stiffMatrix_(1,1) = e_/(1.0 - nu_*nu_);
    stiffMatrix_(0,1) = stiffMatrix_(1,0) = (nu_*e_)/(1.0 - nu_*nu_);
    stiffMatrix_(2,2) = (0.5*e_)/(1.0 + nu_);
  }

  else if ( anmodel_ == "PLANE_STRAIN" )
  {
    const double d = (1.0 + nu_) * (1.0 - 2.0*nu_);

    stiffMatrix_(0,0) = stiffMatrix_(1,1) = e_*(1.0 - nu_)/d;
    stiffMatrix_(0,1) = stiffMatrix_(1,0) = e_*nu_/d;
    stiffMatrix_(2,2) = 0.5*e_/(1.0 + nu_);
  }

  else if ( anmodel_ == "SOLID" )
  {
    const double d = (1.0 + nu_) * (1.0 - 2.0*nu_);

    stiffMatrix_(0,0) = stiffMatrix_(1,1) =
    stiffMatrix_(2,2) = e_*(1.0 - nu_)/d;
    stiffMatrix_(0,1) = stiffMatrix_(1,0) =
    stiffMatrix_(0,2) = stiffMatrix_(2,0) =
    stiffMatrix_(1,2) = stiffMatrix_(2,1) = e_*nu_/d;
    stiffMatrix_(3,3) = stiffMatrix_(4,4) =
    stiffMatrix_(5,5) = 0.5*e_/(1.0 + nu_);
  }

  else
    throw Error ( JEM_FUNC, "Unexpected analysis model: " + anmodel_ );
}

//-----------------------------------------------------------------------
//  computeThermalStrains_ 
//-----------------------------------------------------------------------

void IsotropicMaterial::computeThermalStrains_

  ( Vector&      strain,
    const double deltaT  )

{
  if ( anmodel_ == "BAR" )
  {
    strain.resize ( 1 );
    strain = 0.0;

    strain[0] = alpha_ * deltaT;
  }
  
  else if ( anmodel_ == "PLANE_STRESS" )
  {
    strain.resize ( 3 );
    strain = 0.0;

    strain[0] = alpha_ * deltaT;
    strain[1] = alpha_ * deltaT;
  }

  else if ( anmodel_ == "PLANE_STRAIN" )
  {
    strain.resize ( 3 );
    strain = 0.0;

    strain[0] = alpha_ * deltaT;
    strain[1] = alpha_ * deltaT;

    System::warn() << "Thermal expansion not correct for plane strain!\n";
  }

  else if ( anmodel_ == "SOLID" )
  {
    strain.resize ( 6 );
    strain = 0.0;
    
    strain[0] = alpha_ * deltaT;
    strain[1] = alpha_ * deltaT;
    strain[2] = alpha_ * deltaT;
  }
}

//-----------------------------------------------------------------------
//  computeSwellingStrains_ 
//-----------------------------------------------------------------------

void IsotropicMaterial::computeSwellingStrains_

  ( Vector&      strain,
    const double conc  )

{
  if ( anmodel_ == "BAR" )
  {
    strain.resize ( 1 );
    strain = 0.0;

    strain[0] = swcoeff_ * conc;
  }
  
  else if ( anmodel_ == "PLANE_STRESS" )
  {
    strain.resize ( 3 );
    strain = 0.0;

    strain[0] = swcoeff_ * conc;
    strain[1] = swcoeff_ * conc;
  }

  else if ( anmodel_ == "PLANE_STRAIN" )
  {
    strain.resize ( 3 );
    strain = 0.0;

    strain[0] = swcoeff_ * conc;
    strain[1] = swcoeff_ * conc;

    //System::warn() << "Swelling not correct for plane strain!\n";
  }

  else if ( anmodel_ == "SOLID" )
  {
    strain.resize ( 6 );
    strain = 0.0;
    
    strain[0] = swcoeff_ * conc;
    strain[1] = swcoeff_ * conc;
    strain[2] = swcoeff_ * conc;
  }
}

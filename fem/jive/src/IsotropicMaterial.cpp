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
#include <jem/base/rtti.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/func/ConstantFunc.h>
#include <jive/util/FuncUtils.h>

#include "IsotropicMaterial.h"

#include "utilities.h"
#include "CtypesUtils.h"

#include <cstdlib>

using namespace jem;
using jem::numeric::matmul;
using jem::numeric::norm2;
using jem::numeric::ConstantFunc;
using jem::io::endl;

using jive::util::FuncUtils;

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
const char* IsotropicMaterial::PARAMS_PROP    = "params";
const char* IsotropicMaterial::PARAM_NAMES_PROP = "names";
const char* IsotropicMaterial::PARAM_VALUES_PROP = "values";

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
  Properties myProps = props.findProps ( myName_ );

  props.get  ( rank_, RANK_PROP );
  props.get  ( anmodel_, ANMODEL_PROP );
  props.find ( alpha_, ALPHA_PROP );
  props.find ( swcoeff_, SWELLING_PROP );

  JEM_PRECHECK ( rank_ >= 1 && rank_ <= 3 );

  Properties paramProps;
  if ( myProps.find ( paramProps, PARAMS_PROP ) ){
  System::out() << paramProps << "\n\n";
    paramProps.get( paramNames_, PARAM_NAMES_PROP );
    paramProps.get( paramValues_, PARAM_VALUES_PROP );
  }

  if ( paramNames_.size() != paramValues_.size() ){
    throw Error ( JEM_FUNC, "params.names and params.values have different sizes" );
  }

  String args = "i, t, x, y, z";
  double argvals[ 5 + paramNames_.size() ];

  for ( int i = 0; i < paramNames_.size(); i++ ){
    args = args + ", " + paramNames_[i];
    argvals[5 + i] = paramValues_[i];
  }

  Ref<Object> eObj;
  Ref<Object> ipFieldObj;
  Ref<Object> ipValueObj;

  STRING_ARRAY_PTR ipFields;
  DOUBLE_ARRAY_PTR ipValues = DOUBLE_ARRAY_PTR();
  int eFieldOffset = -1;
  int xFieldOffset = -1;
  int yFieldOffset = -1;

  props.get( eObj, E_PROP );

  if ( isInstance<String>( eObj ) && toValue<String>(eObj) == "globdat" ){
    e_ = nullptr;

    globdat.get(ipFieldObj, "input.ipfields");
    globdat.get(ipValueObj, "input.ipvalues");
    ipFields = ObjectTraits<STRING_ARRAY_PTR>::toValue(ipFieldObj);
    ipValues = ObjectTraits<DOUBLE_ARRAY_PTR>::toValue(ipValueObj);

    System::out() << "Looping over ipFields\n\n";
    for ( int i = 0; i < ipFields.shape[0]; ++i ){
      if ( strcmp(ipFields.ptr[i], "e") == 0 ){
        eFieldOffset = i * ipValues.shape[1];
      } else if ( strcmp(ipFields.ptr[i], "xcoord") == 0 ){
        xFieldOffset = i * ipValues.shape[1];
      } else if ( strcmp(ipFields.ptr[i], "ycoord") == 0 ){
        yFieldOffset = i * ipValues.shape[1];
      }
    }

    if ( eFieldOffset < 0 || xFieldOffset < 0 || yFieldOffset < 0 ){
      throw Error ( JEM_FUNC, "Field not found" );
    }
  } else {
    FuncUtils::configFunc ( e_, args, E_PROP, myProps, globdat );
  }

  if ( rank_ == 1 ){
    FuncUtils::configFunc ( area_, args, AREA_PROP, myProps, globdat );
  } else {
    FuncUtils::configFunc ( nu_, args, NU_PROP, myProps, globdat );
  }

  for ( idx_t ipoint = 0; ipoint < pointCount(); ipoint++ ){
    for ( idx_t ir = 0; ir < rank_; ir++ ){
      argvals[ir + 2] = ipCoords_(ir, ipoint);
    }

    if ( e_ ){
      es_[ipoint] = e_->getValue(argvals);
    } else {
      double xdiff = ipValues.ptr[ipoint + xFieldOffset] - ipCoords_(0, ipoint);
      double ydiff = ipValues.ptr[ipoint + yFieldOffset] - ipCoords_(1, ipoint);

      if ( fabs(xdiff) > 1e-8 || fabs(ydiff) > 1e-8 ){
        throw Error ( JEM_FUNC, "Incorrect integration point location assumed" );
      }

      es_[ipoint] = ipValues.ptr[ipoint + eFieldOffset];
    }

    if ( rank_ == 1 ){
      areas_[ipoint] = area_->getValue(argvals);
    } else {
      nus_[ipoint] = nu_->getValue(argvals);
    }
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void IsotropicMaterial::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const
{
  FuncUtils::getConfig ( conf, e_, E_PROP );

  if ( rank_ == 1 ){
    FuncUtils::getConfig ( conf, area_, AREA_PROP );
  } else {
    FuncUtils::getConfig ( conf, nu_, NU_PROP );
  }

  conf.set   ( ANMODEL_PROP, anmodel_ );
  conf.set   ( ALPHA_PROP, alpha_ );
  conf.set   ( SWELLING_PROP, swcoeff_ );
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
  es_.resize ( npoints );
  areas_.resize ( npoints );
  nus_.resize ( npoints );

  iPointConc_.resize   ( npoints );
  iPointDeltaT_.resize ( npoints );

  es_ = 0.0;
  nus_ = 0.0;
  areas_ = 1.0;

  iPointConc_   = 0.0;
  iPointDeltaT_ = 0.0;

  ipCount_ = npoints;
}


void IsotropicMaterial::createIntPoints

  ( const Matrix& ipCoords )

{
  idx_t rank = ipCoords.size(0);
  idx_t npoints = ipCoords.size(1);

  ipCoords_.resize ( rank, npoints );
  ipCoords_ = ipCoords;

  createIntPoints( npoints );
}

//-----------------------------------------------------------------------
//  pointCount
//-----------------------------------------------------------------------


idx_t IsotropicMaterial::pointCount() const

{
  if ( ipCount_ <= 0 ){
    throw Error ( JEM_FUNC, "No integration points found" );
  }

  return ipCount_;
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
  stiffAtPoint(stiff, ipoint);

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

  matmul ( stress, stiff, mechStrain );
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
  idx_t strCount = STRAIN_COUNTS[rank_];

  Vector strvec ( strCount );
  Matrix stiff ( strCount, strCount );

  update ( stiff, strvec, strain, ipoint );

  if ( anmodel_ == "PLANE_STRAIN" )
  {
    stress[0] = strvec[0];
    stress[1] = strvec[1];
    stress[2] = strvec[2];
    stress[3] = nus_[ipoint] * ( strvec[0] + strvec[1] );
  }
  else
    stress = strvec;
}

//-----------------------------------------------------------------------
//  stiffAtPoint 
//-----------------------------------------------------------------------

void IsotropicMaterial::stiffAtPoint

  ( Matrix&       stiff,
    const idx_t   ipoint )
{
  // reset stiffness matrix beforehand
  stiff = 0.0;

  if ( anmodel_ == "BAR" )
  {
    stiff(0,0) = es_[ipoint] * areas_[ipoint];
  }
  
  else if ( anmodel_ == "PLANE_STRESS" )
  {
    double e = es_[ipoint];
    double nu = nus_[ipoint];

    stiff(0,0) = stiff(1,1) = e / (1.0 - nu * nu);
    stiff(0,1) = stiff(1,0) = (nu * e) / (1.0 - nu * nu);
    stiff(2,2) = (0.5 * e) / (1.0 + nu);
  }

  else if ( anmodel_ == "PLANE_STRAIN" )
  {
    double e = es_[ipoint];
    double nu = nus_[ipoint];
    const double d = (1.0 + nu) * (1.0 - 2.0 * nu);

    stiff(0,0) = stiff(1,1) = e * (1.0 - nu) / d;
    stiff(0,1) = stiff(1,0) = e * nu / d;
    stiff(2,2) = 0.5 * e / (1.0 + nu);
  }

  else if ( anmodel_ == "SOLID" )
  {
    double e = es_[ipoint];
    double nu = nus_[ipoint];
    const double d = (1.0 + nu) * (1.0 - 2.0 * nu);

    stiff(0,0) = stiff(1,1) =  stiff(2,2) = e * (1.0 - nu) / d;
    stiff(0,1) = stiff(1,0) = stiff(0,2) = stiff(2,0) = stiff(1,2) = stiff(2,1) = e * nu / d;
    stiff(3,3) = stiff(4,4) = stiff(5,5) = 0.5 * e / (1.0 + nu);
  }

  else
    throw Error ( JEM_FUNC, "Unexpected analysis model: " + anmodel_ );
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

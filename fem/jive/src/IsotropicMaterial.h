/*
 *  TU Delft / Knowledge Centre WMC
 *
 *  Iuri Barcelos, August 2015
 *
 *  Simple isotropic material class to compute stiffness and stresses.
 *
 */

#ifndef ISOTROPIC_MATERIAL_H
#define ISOTROPIC_MATERIAL_H

#include "Material.h"

//-----------------------------------------------------------------------
//   class IsotropicMaterial
//-----------------------------------------------------------------------

class IsotropicMaterial : public virtual Material
{
 public:

  static const char*     E_PROP;
  static const char*     NU_PROP;
  static const char*     RANK_PROP;
  static const char*     ANMODEL_PROP;
  static const char*     AREA_PROP;
  static const char*     ALPHA_PROP;
  static const char*     SWELLING_PROP;

  explicit               IsotropicMaterial

    ( const String&        name,
      const Properties&    props,
      const Properties&    conf,
      const Properties&    globdat );

  virtual void           configure

    ( const Properties&    props,
      const Properties&    globdat );

  virtual void           getConfig

    ( const Properties&    conf,
      const Properties&    globdat )   const;

  virtual void           update

    ( Matrix&              stiff,
      Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  virtual void           stressAtPoint

    ( Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  virtual void           stiffAtPoint

    ( Vector&              stiffvec,
      const idx_t          ipoint );

  virtual double         strengthAtPoint

    ( const idx_t          ipoint );

  virtual void           addTableColumns

    ( IdxVector&           jcols,
      XTable&              table,
      const String&        name );
  
  virtual bool           hasThermal ();

  virtual bool           hasSwelling ();

  virtual void           createIntPoints

    ( const idx_t           npoints );

  virtual void           setConc

    ( const idx_t           ipoint,
      const double          conc    );

  virtual void           setDeltaT

    ( const idx_t           ipoint,
      const double          deltaT  );

 protected:

  virtual               ~IsotropicMaterial();

 protected:

  idx_t                   rank_;
  double                  e_;
  double                  nu_;
  double                  area_;
  double                  alpha_;
  double                  swcoeff_;
  Matrix                  stiffMatrix_;

  Vector                  iPointConc_;
  Vector                  iPointDeltaT_;

  String                  anmodel_;

 protected:
  
  void                    computeStiffMatrix_     ( );

  void                    computeThermalStrains_  
  
    ( Vector&               strain,
      const double          deltaT  );

  void                   computeSwellingStrains_

    ( Vector&               strain,
      const double          conc    );

};

#endif

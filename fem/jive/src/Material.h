/*
 *  TU Delft / Knowledge Centre WMC
 *
 *  Iuri Barcelos, August 2015
 *
 *  Simple base class for materials.
 *
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/XTable.h>

using jem::System;
using jem::idx_t;
using jem::Ref;
using jem::NamedObject;
using jem::String;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::BoolVector;
using jive::StringVector;
using jive::util::XTable;


//-----------------------------------------------------------------------
//   class Material
//-----------------------------------------------------------------------

class Material : public NamedObject
{
 public:
                         Material

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

  virtual void           update

    ( Matrix&              stiff,
      Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  virtual void           stressAtPoint

    ( Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  virtual void           addTableColumns
  
    ( IdxVector&           jcols,
      XTable&              table,
      const String&        name );

  idx_t                   getHistoryCount   () const;

  StringVector            getHistoryNames   () const;

  void                    getHistoryNames

    ( const StringVector&   hnames ) const;

  virtual void           getHistory

    ( Vector&              hvals,
      const idx_t          mpoint );

  virtual void           setHistory

    ( const Vector&        hvals,
      const idx_t          mpoint );

  virtual void           copyHistory

    ( const idx_t          master,
      const idx_t          slave );

  virtual bool           hasThermal ();

  virtual bool           hasSwelling ();

  virtual bool           hasCrackBand ();

  virtual void           createIntPoints

    ( const idx_t          npoints );

  virtual void           createIntPoints

    ( const Matrix&        ipCoords );

  virtual void           setCharLength

    ( const idx_t          ipoint,
      const double         le );

  virtual void            setConc

    ( const idx_t           ipoint,
      const double          conc    );

  virtual void            setDeltaT

    ( const idx_t           ipoint,
      const double          deltaT  );

  virtual void            commit ();
  
  virtual void            checkCommit
  
    ( const Properties&     params  );

  virtual void            commitOne
  
    ( const idx_t           ipoint );

  virtual void            cancel ();

  virtual String          getFileName

    ( const idx_t           ipoint ) const;

  virtual double          getDissipation

    ( const idx_t           ipoint ) const;

  virtual double          getDissipationGrad

    ( const idx_t           ipoint ) const;

  virtual idx_t           pointCount () const;

  virtual bool           isLoading 
    
    ( const idx_t             ipoint ) const;

  virtual bool           wasLoading
    
    ( const idx_t             ipoint ) const;

 virtual bool            isInelastic

    ( const idx_t             ipoint ) const;

  bool                    despair ();

  void                    endDespair ();

  virtual void            writeState ();

 protected:

                       ~Material ();

  idx_t                ipCount_;
  Matrix               ipCoords_;
  StringVector         historyNames_;
  bool                 desperateMode_;
  BoolVector           hasSwitched_;
  BoolVector           useSecant_;
};

//-----------------------------------------------------------------------
//   newMaterial
//-----------------------------------------------------------------------

Ref<Material>  newMaterial

    ( const String&       name,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );

#endif

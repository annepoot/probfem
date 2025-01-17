
#include "modules.h"

#include <jive/fem/InputModule.h>
#include <jive/app/ModuleFactory.h>


//-----------------------------------------------------------------------
//   declareModules
//-----------------------------------------------------------------------


void declareModules ()
{
  declareGlobdatInputModule ();
  declareGmshInputModule    ();
  declareGroupInputModule   ();
  declareInputModule        ();
}

void declareInputModule ()
{
  using jive::app::ModuleFactory;
  using jive::fem::InputModule;

  ModuleFactory::declare ( "Input",
                         & InputModule::makeNew );
}


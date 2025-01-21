#include <jem/base/Object.h>
#include <jem/base/Throwable.h>
#include <jem/mp/UniContext.h>
#include <jive/app/Module.h>
#include <jive/app/ProgramArgs.h>
#include <jive/util/ObjectConverter.h>

#include "ExposedApplication.h"

typedef jem::mp::Context MPContext;

using namespace jem;
using jem::mp::UniContext;
using jive::app::Module;
using jive::app::ProgramArgs;


//=======================================================================
//   class ExposedApplication
//=======================================================================

//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


int ExposedApplication::exec

  ( int                argc,
    char**             argv,
    ModuleConstructor  ctor,
    Properties         globdat )

{
  String  phase  = "initialization";
  int     result = 1;

  try
  {
    Properties props;
    loadProperties  ( props, argc, argv  );
    result = execProps( props, ctor, globdat );
  }
  catch ( const jem::Throwable& ex )
  {
    String  where = ex.where ();

    if ( ! where.size() && argc > 0 )
    {
      where = argv[0];
    }

    printError ( phase, ex.name(),
                 where, ex.what(), ex.getStackTrace() );
  }
  catch ( ... )
  {
    printError ( phase );
  }

  return result;
}


int ExposedApplication::execProps

  ( Properties         props,
    ModuleConstructor  ctor,
    Properties         globdat )

{
  using jive::util::ObjConverter;

  String  phase  = "initialization";
  int     result = 1;

  try
  {
    Ref<ProgramArgs>  args;
    Ref<MPContext>    mpx;
    Ref<Module>       mod;

    Properties        conf;

    initSigHandlers ();

    args = newInstance<ProgramArgs> ();
    mpx  = newInstance<UniContext>  ();

    initSystem  ( *mpx, conf,  props );
    initGlobdat ( conf, props, globdat, mpx, args );

    props.setConverter ( newInstance<ObjConverter>( globdat ) );

    mod = ctor ();

    if ( ! mod )
    {
      return 0;
    }

    runLoop ( phase, *mod, conf, props, globdat );

    result = 0;
  }
  catch ( const jem::Throwable& ex )
  {
    String  where = ex.where ();

    printError ( phase, ex.name(),
                 where, ex.what(), ex.getStackTrace() );
  }
  catch ( ... )
  {
    printError ( phase );
  }

  return result;
}



#ifndef JIVE_APP_EXPOSEDAPPLICATION_H
#define JIVE_APP_EXPOSEDAPPLICATION_H

#include <jem/util/Properties.h>
#include <jive/app/Application.h>

using jem::util::Properties;
using jive::app::Application;

//-----------------------------------------------------------------------
//   class ExposedApplication
//-----------------------------------------------------------------------


class ExposedApplication : Application
{
 public:

  static int                exec

    ( int                     argc,
      char**                  argv,
      ModuleConstructor       ctor,
      Properties              outdat );
};

#endif



#include "models.h"


//-----------------------------------------------------------------------
//   declareModels
//-----------------------------------------------------------------------


void declareModels ()
{
  declareDirichletModel();
  declareElasticModel();
  declareLaplaceModel();
//  declareLoadDispModel();
  declareNeumannModel();
//  declareSolidModel();
  declareSpringModel();
}



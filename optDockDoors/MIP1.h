#pragma once
#include "common.h"

// classe contenente la formulazione da internet
class MIP1
{
   private:
      int computeTimeRequest(int i, int j);

   public:
      vector<vector<int>> cost,q;
      vector<int> cap;
      double forkLiftSpeed;

      void run_MIP1();
      int populateTableau(CPXENVptr env, CPXLPptr lp);
      string var(const string& base, int i, int j); 
      int model();
};


#pragma once
#include "common.h"

// classe contenente la formulazione internet
class MIP2
{
   private:
      vector<string> colnames; // i nomi delle variabili

      int computeTimeRequest(int i, int j);
      tuple<int,int,int,float,float,double,double> callCPLEX(int timeLimit, bool isVerbose);
      int populateTableau(CPXENVptr env, CPXLPptr lp, int bigM);
      string var(const string& base, int i, int j); 
      void checkSol();

   public:
      vector<vector<int>> cost,p;
      double forkLiftSpeed;
      std::vector<std::vector<int>> sol;

      void run_MIP2(int timeLimit, bool isVerbose);
};


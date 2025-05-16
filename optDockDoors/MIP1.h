#pragma once
#include "common.h"

// classe contenente la formulazione GAP
class MIP1
{
   private:
      vector<string> colnames; // i nomi delle variabili

      int computeTimeRequest(int i, int j);
      tuple<int,int,int,float,float,double,double> callCPLEX(int timeLimit, bool isVerbose);
      int populateTableau(CPXENVptr env, CPXLPptr lp);
      string var(const string& base, int i, int j); 
      void checkSol();

   public:
      vector<vector<int>> cost,q;
      vector<int> cap;
      double forkLiftSpeed;
      std::vector<std::vector<int>> sol;

      void run_MIP1(int timeLimit, bool isVerbose);
      int model();
};


#include "MIP1.h"

// classe contenente la forulazione da internet
void MIP1::run_MIP1()
{
   cout << m << endl;
   // Initialize CPLEX environment and problem
   CPXENVptr env = CPXopenCPLEX(NULL);
   if (env==NULL)
   {
      cerr<<"Failed to open CPLEX environment."<<endl;
      return;
   }
   CPXLPptr lp = CPXcreateprob(env, NULL, "MIP1");
   if (lp==NULL)
   {
      cerr<<"Failed to create CPLEX problem."<<endl;
      CPXcloseCPLEX(&env);
      return;
   }
   // Set parameters for CPLEX
   CPXsetintparam(env, CPXPARAM_MIP_Tolerances_MIPGap, 0);
   CPXsetintparam(env, CPXPARAM_MIP_Limits_Solutions, 1);
   // Define the model here...
   // Solve the model
   int status = CPXmipopt(env, lp);
   if (status)
   {
      cerr<<"Failed to optimize the model."<<endl;
      CPXfreeprob(env, &lp);
      CPXcloseCPLEX(&env);
      return;
   }
   // Retrieve solution and write to file
   solFile = "solution.txt";
   ofstream outFile(solFile);
   if (!outFile.is_open())
   {
      cout<<"Failed to open solution file: "<<solFile<<endl;
      CPXfreeprob(env, &lp);
      CPXcloseCPLEX(&env);
      return;
   }
   // Write solution to file...

   outFile.close();

   // Clean up
   CPXfreeprob(env, &lp);
   CPXcloseCPLEX(&env);
   return;
}
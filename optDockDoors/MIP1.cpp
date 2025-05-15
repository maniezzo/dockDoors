#include "MIP1.h"
// classe contenente la forulazione da internet


// The tableu for the basic GAP, non scenario case.
int MIP1::populateTableau(CPXENVptr env, CPXLPptr lp)
{  int status,numrows,numcols,numnz;
   int i,j,currMatBeg;
   vector<double> obj;
   vector<double> lb;
   vector<double> ub;
   vector<string> colname;
   vector<int>    rmatbeg;
   vector<int>    rmatind;
   vector<double> rmatval;
   vector<double> rhs;
   vector<char>   sense;
   vector<string> rowname;

   status = CPXchgobjsen(env, lp, CPX_MIN);  // Problem is minimization

   // ------------------------------------------------------ variables section
   
   // Create the columns for x variables
   numcols = 0;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
      {  obj.push_back(cost[i][j]); numcols++;  
         lb.push_back(0.0);  
         ub.push_back(1.0); 
         colname.push_back("x"+to_string(i)+"_"+to_string(j));
      }

   char** cname = new char* [colname.size()];
   for (int index = 0; index < colname.size(); index++)
      cname[index] = const_cast<char*>(colname[index].c_str());
   status = CPXnewcols(env, lp, numcols, &obj[0], &lb[0], &ub[0], NULL, cname);
   delete[] cname;

   if (status)  cout << "ERROR" << endl;

   // ------------------------------------------------------ constraints section

   // num assignments x constraints.
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(j=0;j<n;j++)
      {  rmatbeg.push_back(currMatBeg);
         rowname.push_back("a"+to_string(j)); numrows++;
         for(i=0;i<m;i++)
         {
            rmatind.push_back(i*n+j); 
            rmatval.push_back(1); 
            numnz++;
         }
         sense.push_back('E');
         rhs.push_back(1.0);
         currMatBeg+=m;
      }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) {
         rname[index] = const_cast<char*>(rowname[index].c_str());
      }
      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }

   // capacity constraints
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(i=0;i<m;i++)
      {  rmatbeg.push_back(currMatBeg);
         rowname.push_back("cap"+to_string(i)); numrows++;
         for(j=0;j<n;j++)
         {
            rmatind.push_back(i*n+j); 
            rmatval.push_back(q[i][j]); 
            numnz++;
         }
         sense.push_back('L');
         rhs.push_back(cap[i]);
         currMatBeg+=n;
      }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) {
         rname[index] = const_cast<char*>(rowname[index].c_str());
      }
      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }


   TERMINATE:
   return (status);
} 

// calcola il tempo necessario per caricare il camion j sull'area i
int MIP1::computeTimeRequest(int i, int j)
{  int k;
   double tload,t=0;

   for(k=0;k<4;k++)
   {  tload = dist[i][k]/forkLiftSpeed;
      t += req[j][k]*tload;
      t += req[i][k] * 150; // da area a camion
   }
   return round(t);
}

// Function to run the MIP1 model
void MIP1::run_MIP1()
{  int i,j;
   cout << m << endl;
   // Initialize CPLEX environment and problem
   CPXENVptr env = CPXopenCPLEX(NULL);
   if (env==NULL)
   {  cout<<"Failed to open CPLEX environment."<<endl;
      return;
   }
   CPXLPptr lp = CPXcreateprob(env, NULL, "MIP1");
   if (lp==NULL)
   {  cout<<"Failed to create CPLEX problem."<<endl;
      CPXcloseCPLEX(&env);
      return;
   }
   // Set parameters for CPLEX
   CPXsetintparam(env, CPXPARAM_MIP_Tolerances_MIPGap, 0);
   CPXsetintparam(env, CPXPARAM_MIP_Limits_Solutions, 1);

   // ---------------------------------------------------- Define the model
   q.resize(m); // Resize to m rows
   for (i = 0; i < m; ++i) 
   {  q[i].resize(n); // Resize each row to n columns
      for (j = 0; j < n; ++j) 
         q[i][j] = computeTimeRequest(i,j);
   }

   cost.resize(m); // Resize to m rows
   for (i = 0; i < m; ++i) 
   {  cost[i].resize(n); // Resize each row to n columns
      for (j = 0; j < n; ++j) 
         cost[i][j] = 1;
   }

   cap.resize(m);
   for(i=0;i<m;++i)
      cap[i] = tmax;

   populateTableau(env, lp);
   int status = CPXwriteprob (env, lp, "probl.lp", NULL);

   // ----------------------------------------------------- Solve the model
   status = CPXmipopt(env, lp);
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

// Helper function to name variables
string MIP1::var(const string& base, int i, int j = -1) 
{  if (j == -1) return base + "_" + to_string(i);
   return base + "_" + to_string(i) + "_" + to_string(j);
}

int MIP1::model() 
{  const int n = 4; // number of jobs
   const int m = 2; // number of machines

   vector<int> p = {4, 3, 2, 5}; // processing times
   vector<int> d = {10, 6, 8, 9}; // due dates

   int BIG_M = 0;
   for (int pj : p) BIG_M += pj;

   ofstream lp("model.lp");

   // Objective
   lp << "Minimize\n obj: ";
   for (int j = 0; j < n; ++j) {
      lp << "+ T_" << j << " ";
   }
   lp << "\nSubject To\n";

   // Each job assigned to one machine
   for (int j = 0; j < n; ++j) {
      lp << " assign_" << j << ": ";
      for (int i = 0; i < m; ++i)
         lp << "+ " << var("x", i, j) << " ";
      lp << "= 1\n";
   }

   // Completion time
   for (int j = 0; j < n; ++j) {
      lp << " complete_" << j << ": "
         << var("C", j) << " - " << var("S", j)
         << " = " << p[j] << "\n";
   }

   // Tardiness
   for (int j = 0; j < n; ++j) {
      lp << " tardiness_" << j << "_1: "
         << var("T", j) << " - " << var("C", j)
         << " >= " << -d[j] << "\n";
      lp << " tardiness_" << j << "_2: "
         << var("T", j) << " >= 0\n";
   }

   // Disjunctive constraints
   for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
         for (int k = 0; k < n; ++k) {
            if (j == k) continue;
            string xij = var("x", i, j);
            string xik = var("x", i, k);
            string sj = var("S", j);
            string sk = var("S", k);
            string zjk = var("z", j, k);

            string tag1 = "precede_" + to_string(i) + "_" + to_string(j) + "_" + to_string(k) + "_1";
            string tag2 = "precede_" + to_string(i) + "_" + to_string(j) + "_" + to_string(k) + "_2";

            // Only enforce if both assigned to same machine
            lp << tag1 << ": "
               << sj << " + " << p[j] << " - " << sk
               << " <= " << BIG_M << " * (2 - " << xij << " - " << xik << " + " << zjk << ")\n";

            lp << tag2 << ": "
               << sk << " + " << p[k] << " - " << sj
               << " <= " << BIG_M << " * (2 - " << xij << " - " << xik << " + 1 - " << zjk << ")\n";
         }
      }
   }

   // Variable declarations
   lp << "Bounds\n";
   for (int j = 0; j < n; ++j) {
      lp << " " << var("S", j) << " >= 0\n";
      lp << " " << var("C", j) << " >= 0\n";
      lp << " " << var("T", j) << " >= 0\n";
   }

   lp << "Binary\n";
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
         lp << " " << var("x", i, j) << "\n";

   for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
         if (j != k)
            lp << " " << var("z", j, k) << "\n";

   lp << "End\n";
   lp.close();

   cout << "Model written to model.lp\n";
   return 0;
}

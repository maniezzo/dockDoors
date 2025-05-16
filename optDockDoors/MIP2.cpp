#include "MIP2.h"
// classe contenente la forulazione internet

// Function to run the MIP2 internet model
void MIP2::run_MIP2(int timeLimit, bool isVerbose)
{  int i,j;

   // ---------------------------------------------------- Define the model
   p.resize(m); // Resize to m rows
   for (i = 0; i < m; ++i) 
   {  p[i].resize(n); // Resize each row to n columns
      for (j = 0; j < n; ++j) 
         p[i][j] = computeTimeRequest(i,j);
   }

   cost.resize(m); // Resize to m rows
   for (i = 0; i < m; ++i) 
   {  cost[i].resize(n); // Resize each row to n columns
      for (j = 0; j < n; ++j) 
         if(i<m-1)
            cost[i][j] = 0;
         else
            cost[i][j] = 1;
   }

   sol.resize(m);

   n = 4; // number of jobs (trucks)
   m = 2; // number of machines (areas)

   p = { {1, 2, 3, 4},   // Row 0
         {5, 6, 7, 8}    // Row 1
       }; // processing times
   vector<int> d = {10, 6, 8, 9}; // due dates

   tuple<int,int,int,float,float,double,double> res = callCPLEX(timeLimit,isVerbose);
   checkSol();

   string strInst = "MIP2";
   int numScen = 0; 
   int nboost = 0; 
   ostringstream osString;
   osString<<"Instance "      << strInst;
   osString<<" num.scen. "    << numScen;
   osString<<" num.boost "    << nboost;
   osString<<" status " <<    get<0>(res);
   osString<<" cur_numcols "<<get<1>(res);
   osString<<" cur_numrows "<<get<2>(res);
   osString<<" zlb "<<        get<3>(res);
   osString<<" objval "<<     get<4>(res);
   osString<<" finalLb "<<    get<5>(res);
   osString<<" total_time "<< get<6>(res)<<endl;
   string outStr = osString.str();
   cout<< fixed << outStr<<endl;

   return;
}

// checks the feasibility of the solution
void MIP2::checkSol()
{  int i,j,k,load;

   for(j=0;j<n;j++)
   {
      for (i=0;i<m;i++)
         for (k=0;k<sol[i].size();k++)
         {  if (sol[i][k]==j)
               goto lFound;
         }
      cout<<"Truck "<<j<<" not assigned to any area"<<endl;
      break;
lFound: continue;
   }
}

// The tableau for the basic GAP, non scenario case.
int MIP2::populateTableau(CPXENVptr env, CPXLPptr lp, int bigM)
{  int status,numrows,numcols,numnz;
   int i,j,k,currMatBeg;
   int startx, startz, starts, startc, startt, startTmax; // inidici inizio di ogni tipo di variabile
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
   // Variabili: x_ij, z_ij, s_j, c_j, t_j, Tmax
   int Tsup = 50000; // massima tardiness possibile

   // Create the columns for x variables, job j assigned to area i
   numcols = 0;
   startx = 0;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
         {  obj.push_back(1);
            lb.push_back(0.0);  
            ub.push_back(1.0); 
            colname.push_back("x"+to_string(i)+"_"+to_string(j));
            numcols++;
         }

   // Create the columns for z variables, i before j
   startz = numcols;
   for(i=0;i<n;i++)
      for(j=0;j<n;j++)
      {  obj.push_back(1);
         lb.push_back(0.0);  
         ub.push_back(1.0); 
         colname.push_back("z"+to_string(i)+"_"+to_string(j));
         numcols++;
      }

   // Create the columns for s variables, start times
   starts = numcols;
   for(j=0;j<n;j++)
   {  obj.push_back(1);
      lb.push_back(0.0);  
      ub.push_back(Tsup); 
      colname.push_back("s"+to_string(j));
      numcols++;
   }

   // Create the columns for c variables, completion times
   startc = numcols;
   for(j=0;j<n;j++)
   {  obj.push_back(1);
      lb.push_back(0.0);  
      ub.push_back(Tsup); 
      colname.push_back("c"+to_string(j));
      numcols++;
   }

   // Create the columns for t variables, tardinesses
   startt = numcols;
   for(j=0;j<n;j++)
   {  obj.push_back(1);
      lb.push_back(0.0);  
      ub.push_back(Tsup); 
      colname.push_back("t"+to_string(j));
      numcols++;
   }

   // Create the columns for Tmax variable, max tardiness
   startTmax = numcols;
   obj.push_back(1);
   lb.push_back(0.0);  
   ub.push_back(Tsup); 
   colname.push_back("Tmax");
   numcols++;

   // I nomi di tutte le variabili, di ogni tipo
   char** cname = new char* [colname.size()];
   for (int index = 0; index < colname.size(); index++)
   {  cname[index] = const_cast<char*>(colname[index].c_str());
      colnames.push_back(colname[index]);
   }
   status = CPXnewcols(env, lp, numcols, &obj[0], &lb[0], &ub[0], NULL, cname);
   if (status)  cout << "ERROR" << endl;
   delete[] cname;

   // ------------------------------------------------------ constraints section

   // assignment x constraints.
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
         rowname.push_back("a"+to_string(j)); 
         numrows++;
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

   // disjunctive constraints sj + pj \leq si + M(3 - xki -xkj - zjk)
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(k=0;k<m;k++)
         for(i=0;i<n;i++)
         {  for(j=0;j<n;j++)
            {
               rmatbeg.push_back(currMatBeg);
               rowname.push_back("d"+to_string(i)+to_string(j)+to_string(k)); 
               numrows++;

               // Mxki
               rmatind.push_back(k*n+i); 
               rmatval.push_back(bigM); 
               numnz++;
               // Mxkj
               rmatind.push_back(k*n+j); 
               rmatval.push_back(bigM); 
               numnz++;
               // Mzij
               rmatind.push_back(startz+i*n+j); 
               rmatval.push_back(bigM); 
               numnz++;
               // -si+sj
               rmatind.push_back(starts+i); 
               rmatval.push_back(-1); 
               numnz++;
               rmatind.push_back(starts+j); 
               rmatval.push_back(-1); 
               numnz++;

               sense.push_back('L');
               rhs.push_back(3*bigM-p[k][i]);
               currMatBeg+=5;
            }
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

// Helper function to name variables
string MIP2::var(const string& base, int i, int j = -1) 
{  if (j == -1) return base + "_" + to_string(i);
   return base + "_" + to_string(i) + "_" + to_string(j);
}

// calcola il tempo necessario per caricare il camion j sull'area i
int MIP2::computeTimeRequest(int i, int j)
{  int k;
   double tload,t=0;

   for(k=0;k<4;k++)
   {  tload = dist[i][k]/forkLiftSpeed;
      t += req[j][k]*tload;
      t += req[i][k] * 150; // da area a camion
   }
   return round(t);
}

// Function to call CPLEX and solve the model
tuple<int,int,int,float,float,double,double> MIP2::callCPLEX(int timeLimit, bool isVerbose)
{  int      solstat;
   double   objval,zlb,lbfinal;
   vector<double> x;
   vector<double> pi;
   vector<double> slack;
   vector<double> dj;
   vector<char>   ctype;  

   int    status = 0;
   int    i,j;
   int    cur_numrows, cur_numcols;
   time_t tstart, tend; 
   double total_time;
   CallbackData data;

   int area=-1,truck;
   int bigM = 0;
   for (i=0;i<m;i++) 
      for(j=0;j<n;j++)
         bigM += p[i][j]; // esagerato

   // Initialize CPLEX environment and problem
   CPXENVptr env = CPXopenCPLEX(NULL);
   CPXLPptr  lp  = CPXcreateprob(env, NULL, "MIP2");
   if (env==NULL)
   {  char  errmsg[CPXMESSAGEBUFSIZE];
      cout << "Could not open CPLEX environment." << endl;
      CPXgeterrorstring(env, status, errmsg);
      cout << errmsg << endl;
      goto TERMINATE;
   }
   if (lp==NULL)
   {  cout<<"Failed to create CPLEX problem."<<endl;
      CPXcloseCPLEX(&env);
      goto TERMINATE;
   }

   // Turn on data checking
   status = CPXsetintparam(env, CPXPARAM_Read_DataCheck,CPX_DATACHECK_WARN);
   if (status) 
   {  cout << "Failure to turn on data checking, error " << status << endl; goto TERMINATE; }

   // Set parameters for CPLEX
   CPXsetintparam(env, CPXPARAM_MIP_Tolerances_MIPGap, 0.001); // Sets a relative tolerance on the gap between the best integer objective and the objective of the best node remaining.
   // CPXsetintparam(env, CPXPARAM_MIP_Limits_Solutions, 1); // Sets the number of MIP solutions to be found before stopping.
   // time limit
   status = CPXsetdblparam(env, CPX_PARAM_TILIM, timeLimit);

   status = populateTableau(env, lp, bigM);
   if (status) 
   {  cout <<"Failed to populate problem." << endl; goto TERMINATE; }

   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);
   cout << "LP model; ncol=" << cur_numcols << " nrows=" << cur_numrows << endl;
   status = CPXwriteprob (env, lp, "probl.lp", NULL);

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   cout<<"Solving LP relaxation..."<<endl;
   status = CPXlpopt(env, lp);
   if (status) 
   {  cout << "Failed to optimize LP." << endl; goto TERMINATE; }

   // save solutions
   for(j=0;j<cur_numcols;j++)
   {  x.push_back(0);   // primal variables
      dj.push_back(0);  // reduced costs
   }

   for (i = 0; i < cur_numrows; i++)
   {  pi.push_back(0);     // dual variables
      slack.push_back(0);  // constraint slacks
   }

   status = CPXsolution(env, lp, &solstat, &objval, &x[0], &pi[0], &slack[0], &dj[0]);
   if (status) 
   {  cout << "Failed to obtain solution." << endl; goto TERMINATE; }

   zlb = objval;
   // Write the output to the screen.
   cout << "Solution status = " << solstat << endl;
   cout << "Solution value  = " << objval << endl;
   //for (i = 0; i < cur_numrows; i++) 
   //   cout << "Row "<< i << ":  Slack = "<< slack[i] <<"  Pi = " << pi[i] << endl;

   //for (j = 0; j < cur_numcols; j++) 
   //   if(x[j] > 0.01)
   //      cout << "Column " << j << ":  Value = " << x[j] <<"  Reduced cost = " << dj[j] << endl;

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MIP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   cout<<"Solving MIP..."<<endl;
   // Now copy the ctype array
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
         ctype.push_back('I');   // x vars
   status = CPXcopyctype(env, lp, &ctype[0]);
   if (status)
   {  cout << "Failed to copy ctype" << endl; goto TERMINATE; }

   // Create an instance of our callback data structure
   data.lastPrintTime = std::chrono::steady_clock::now(); // Initialize the timer

   // Set the callback function
   status = CPXsetinfocallbackfunc (env, myCallbackFunction, &data);
   if ( status ) 
   {  cout << "Failed to set callback function." << std::endl;
      CPXfreeprob(env, &lp);
      CPXcloseCPLEX(&env);
      return make_tuple<int,int,int,float,float,double,double>(0,0,0,0,0,0,0);
   }

   // ---------------------------- Optimize to integrality
   tstart = clock();
   status = CPXmipopt(env, lp);
   if (status) 
   {  cout << "Failed to optimize MIP" << endl; goto TERMINATE; }
   tend = clock();
   total_time = (double)( tend - tstart )/(double)CLK_TCK ;
   cout << "Elapsed time :" << total_time << endl;

   solstat = CPXgetstat(env, lp);
   cout << "Solution status = " << solstat << endl;

    status = CPXgetobjval(env, lp, &objval);
   if (status) 
   {  cout << "No MIP objective value available.  Exiting..." << endl; goto TERMINATE; }

   status = CPXgetbestobjval(env, lp, &lbfinal);
   if (status) 
   {  cout << "Could not get a lower bound.  Exiting..." << endl; goto TERMINATE; }

   cout << "Solution value  = " << objval << endl;
   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);

   status = CPXgetx(env, lp, &x[0], 0, cur_numcols - 1);
   if (status) 
   {  cout << "Failed to get optimal integer x." << endl; goto TERMINATE; }

   status = CPXgetslack(env, lp, &slack[0], 0, cur_numrows - 1);
   if (status) 
   {  cout << "Failed to get optimal slack values." << endl; goto TERMINATE; }

   //for (i = 0; i < cur_numrows; i++) 
   //   cout << "Row " << i << ":  Slack = " << slack[i] << endl;
   for (j = 0; j<cur_numcols; j++)
      if (x[j]>0.01)
      {  if (isVerbose)
            cout<<"Column "<<j<<" var "<<colnames[j]<<":  Value = "<<x[j]<<endl;
         // Skip first character x
         string rest = colnames[j].substr(1); 
         size_t underscorePos = rest.find('_');
         int num1 = stoi(rest.substr(0, underscorePos));
         int num2 = stoi(rest.substr(underscorePos + 1));
         if(num1!=area)
         {  area = num1;
            cout<<endl<<"Area,"<< area << ",trucks,";
         }
         cout << num2 << ",";
         sol[area].push_back(num2);
      }

   // Finally, write a copy of the problem to a file
   if(cur_numcols < 200)
   {  status = CPXwriteprob(env,lp,"probl.lp",NULL);
      if (status) 
      {  cout << "Failed to write model to disk." << endl; goto TERMINATE; }
   }

TERMINATE:
   // Clean up
   CPXfreeprob(env, &lp);
   CPXcloseCPLEX(&env);
   tuple<int,int,int,float,float,double,double> res = make_tuple(status,cur_numcols,cur_numrows,zlb,objval,lbfinal,total_time);
   return res;
}

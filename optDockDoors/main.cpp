#include "common.h"
#include "MIP1.h"
#include "MIP2.h"
#include "MIP3.h"

// globals
int m;   // number of areas
int n;   // number of trucks
int tmax;// capacity
vector<vector<int>> dist;
vector<vector<int>> req;
vector<vector<int>> tw;
string solFile; // global

int readInstance(string distanceFile, string requestsFile, string tWindowsFile)
{  string line;
   int cont;

   // read distances among areas
   ifstream file(distanceFile);
   if (!file.is_open()) 
   {  cout << "Failed to open file: " << distanceFile << endl;
      return 1;
   }

   getline(file, line); // skip header
   m = cont = 0;
   while (getline(file, line)) 
   {  vector<int> row;
      stringstream ss(line);
      string cell;

      while (getline(ss, cell, ',')) 
         row.push_back(stoi(cell));

      dist.push_back(row);
      m++; 
   }
   file.close();

   // Print distance data to verify
   cout<<"Distance matrix:"<<endl;
   for (const auto& row : dist) 
   {  cout << cont << ") ";
      for (int val : row)
         cout << val << " ";
      cout << endl;
      cont++;
   }

   // read transportation requests
   ifstream fileReq(requestsFile);
   if (!fileReq.is_open()) 
   {  cout << "Failed to open file: " << requestsFile << endl;
      return 1;
   }

   n = cont = 0;
   getline(fileReq, line); // skip header
   while (getline(fileReq, line)) 
   {  vector<int> row;
      stringstream ss(line);
      string cell;

      while (getline(ss, cell, ',')) 
         row.push_back(stoi(cell));

      req.push_back(row);
      n++;
   }
   fileReq.close();

   // Print request data to verify
   cout<<"Request matrix:"<<endl;
   for (const auto& row : req) 
   {  cout << cont << ") ";
      for (int val : row)
         cout << val << " ";
      cout << endl;
      cont++;
   }

   // read time windows
   ifstream fileTW(tWindowsFile);
   if (!fileTW.is_open()) 
   {  cout << "Failed to open file: " << tWindowsFile << endl;
      return 1;
   }
   
   cont = 0;
   getline(fileTW, line); // skip header
   while (getline(fileTW, line)) 
   {  vector<int>  row;
      stringstream ss(line);
      string cell;

      while (getline(ss, cell, ',')) 
         row.push_back(stoi(cell));

      tw.push_back(row);
   }
   fileTW.close();

   // Print time windows data to verify
   cout<<"Time windows matrix:"<<endl;
   for (const auto& row : tw) 
   {  cout << cont << ") ";
      for (int val : row)
         cout << val << " ";
      cout << endl;
      cont++;
   }
   return 0;
}

// callback, returns lower and upper bounds
int CPXPUBLIC myCallbackFunction(CPXCENVptr env, void* cbdata, int wherefrom, void* cbhandle)
{  int status=0,numsec;
   double bestObjVal=-1, incumbObjVal=-1;
   // Cast the callback handle to our custom data structure
   CallbackData *data = static_cast<CallbackData*>(cbhandle);

   // Get the current time
   auto currentTime = chrono::steady_clock::now();

   // Calculate time elapsed since the last print
   chrono::duration<double> elapsedTime = currentTime - data->lastPrintTime;

   // Check if numsec seconds have passed
   numsec = 600;
   if (elapsedTime.count() >= numsec) 
   {  status = CPXgetcallbackinfo (env, cbdata, wherefrom, CPX_CALLBACK_INFO_BEST_INTEGER, &bestObjVal);
      if ( status ) 
      {  cout<<"error " << status << " in CPXgetcallbackinfo"<<endl;
         status = 1;
         goto TERMINATE;
      }

      status = CPXgetcallbackinfo (env, cbdata, wherefrom, CPX_CALLBACK_INFO_BEST_REMAINING, &incumbObjVal);
      if ( status ) 
      {  cout<<"error " << status << " in CPXgetcallbackinfo"<<endl;
         status = 1;
         goto TERMINATE;
      }

      // Print the bounds (if both were retrieved)
      cout << "-----> " << " elapsed " << elapsedTime.count() << " secs: zub: " << bestObjVal << ", zlb: " << incumbObjVal << endl;

      // Reset the timer
      data->lastPrintTime = currentTime;
      // Open the file in append mode (std::ios::app)
      ofstream outFile(solFile,std::ios::app);
      if (!outFile)
         cout<<"Error opening output file: "<<solFile<<endl;
      else
      {  outFile << "elapsed " << elapsedTime.count() << " zlb " << incumbObjVal << " zub " << bestObjVal << endl;
         outFile.close();
      }
   }
   TERMINATE:
   return status; // Zero return indicates success
}

int main()
{  string requestsFile,distanceFile,tWindowsFile;
   string line,inst;
   stringstream ss;

   MIP1 M1;
   MIP2 M2;
   MIP3 M3;

   srand(995);
   //srand(time(NULL));

   ifstream infile;
   cout<<"Opening config.json"<<endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open("config.json");

   std::stringstream buffer;
   buffer << infile.rdbuf();
   line = buffer.str();
   infile.close();
   json::Value JSV = json::Deserialize(line);

   requestsFile   = JSV["requestsFile"];
   distanceFile   = JSV["distanceFile"];
   tWindowsFile   = JSV["tWindowsFile"];
   string solFile = JSV["solFile"];
   tmax           = JSV["tmax"];         // tempo disponibile caricamento camion, in secondi. Capacità aree
   int TimeLimit  = JSV["TimeLimit"];    // CPLEX time limit
   double epsCost = JSV["epsCost"];      // costo ogni infeasibility
   double forkLiftSpeed = JSV["forkLiftSpeed"]; // velocità carrelli
   bool isVerbose = JSV["isVerbose"];

   readInstance(distanceFile,requestsFile,tWindowsFile);
   M1.forkLiftSpeed = forkLiftSpeed;
   //M1.run_MIP1(TimeLimit,isVerbose);
   M2.forkLiftSpeed = forkLiftSpeed;
   M2.run_MIP2(TimeLimit,isVerbose);
   M3.forkLiftSpeed = forkLiftSpeed;
   M3.run_MIP3(TimeLimit,isVerbose);
}

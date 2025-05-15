#include "common.h"
#include "MIP1.h"

// globals
int m;   // number of areas
int n;   // number of trucks
int tmax;// capacity
vector<vector<int>> dist;
vector<vector<int>> req;
string solFile; // global

int readInstance(string distanceFile, string requestsFile)
{
   string line;

   // read distances among areas
   ifstream file(distanceFile);
   if (!file.is_open()) 
   {  cout << "Failed to open file: " << distanceFile << endl;
      return 1;
   }

   getline(file, line); // skip header
   m = 0;
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

   // Print the data to verify
   for (const auto& row : dist) 
   {  for (int val : row)
         cout << val << " ";
      cout << endl;
   }

   // read transportation requests
   ifstream fileReq(requestsFile);
   if (!fileReq.is_open()) 
   {  cout << "Failed to open file: " << requestsFile << endl;
      return 1;
   }

   n = 0;
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

   // Print the data to verify
   for (const auto& row : req) 
   {  for (int val : row)
      cout << val << " ";
      cout << endl;
   }

   return 0;
}

int main()
{  string requestsFile,distanceFile;
   string line,inst;
   stringstream ss;

   MIP1 M1;

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
   string solFile = JSV["solFile"];
   tmax           = JSV["tmax"];         // tempo disponibile caricamento camion, in secondi. Capacità aree
   int TimeLimit  = JSV["TimeLimit"];    // CPLEX time limit
   double epsCost = JSV["epsCost"];      // costo ogni infeasibility
   double forkLiftSpeed = JSV["forkLiftSpeed"]; // velocità carrelli
   bool isVerbose = JSV["isVerbose"];

   readInstance(distanceFile,requestsFile);
   M1.forkLiftSpeed = forkLiftSpeed;
   M1.run_MIP1();
   M1.model();
}

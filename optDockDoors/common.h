#ifndef COMMON_H
#define COMMON_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <ilcplex/cplex.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <tuple>
#include <algorithm>   // sort
#include <time.h>
#include <chrono>
#include "json.h"

using namespace std;

extern int m;   // number of areas
extern int n;   // number of trucks
extern vector<vector<int>> dist;
extern vector<vector<int>> req;
extern string solFile; // global

// Structure to hold callback data, including timing information
struct CallbackData {
   std::chrono::steady_clock::time_point lastPrintTime; // Last time bounds were printed
};

// returns lower and upper bounds
int CPXPUBLIC myCallbackFunction(CPXCENVptr env, void *cbdata, int wherefrom, void *cbhandle);

#endif // COMMON_H
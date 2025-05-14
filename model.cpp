#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

string var(const string& base, int i, int j = -1) {
    if (j == -1) return base + "_" + to_string(i);
    return base + "_" + to_string(i) + "_" + to_string(j);
}

int main() {
    const int n = 4; // number of jobs
    const int m = 2; // number of machines

    vector<int> p = {4, 3, 2, 5}; // processing times
    vector<int> d = {10, 6, 8, 9}; // due dates

    int BIG_M = 0;
    for (int pj : p) BIG_M += pj;

    ofstream lp("model.lp");

    // Objective
    lp << "Minimize\n obj: ";
    for (int j = 0; j < n; ++j)
        lp << "+ T_" << j << " ";
    lp << "\nSubject To\n";

    // Each job assigned to exactly one machine
    for (int j = 0; j < n; ++j) {
        lp << " assign_" << j << ": ";
        for (int i = 0; i < m; ++i)
            lp << "+ " << var("x", i, j) << " ";
        lp << "= 1\n";
    }

    // Completion times: C_j = S_j + p_j
    for (int j = 0; j < n; ++j)
        lp << " complete_" << j << ": " << var("C", j) << " - " << var("S", j) << " = " << p[j] << "\n";

    // Tardiness: T_j >= C_j - d_j
    for (int j = 0; j < n; ++j)
        lp << " tardiness_" << j << ": " << var("T", j) << " - " << var("C", j) << " >= " << -d[j] << "\n";

    // Disjunctive constraints (single direction using z_jk)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (j == k) continue;

                string constraint_name = "order_" + to_string(i) + "_" + to_string(j) + "_" + to_string(k);
                string sj = var("S", j);
                string sk = var("S", k);
                string xij = var("x", i, j);
                string xik = var("x", i, k);
                string zjk = var("z", j, k);

                lp << constraint_name << ": "
                   << sj << " + " << p[j] << " - " << sk
                   << " <= " << BIG_M << " * (3 - " << xij << " - " << xik << " - " << zjk << ")\n";
            }
        }
    }

    // Bounds
    lp << "Bounds\n";
    for (int j = 0; j < n; ++j) {
        lp << " " << var("S", j) << " >= 0\n";
        lp << " " << var("C", j) << " >= 0\n";
        lp << " " << var("T", j) << " >= 0\n";
    }

    // Binary variables
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

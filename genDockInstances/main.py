# genera istanze di dock scheduling partendo dalle previsioni dei num pallet
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import time, pulp
import copy

# distanze in metri, poi in secondi
def computeDistances():
   numBays = 31
   df = pd.DataFrame(columns=['tf', 'tb', 'ta', 'tx'])

   coordfast = (40,20)
   coordbulk = (160,60)
   coordasrs = (40,100)
   coordxdock = (100,0)

   x = 20
   y = 10
   for i in range(numBays):
      y += (2+3*(i%2))
      tf = (30 + 2*np.sqrt((x-coordfast[0])**2 + (y-coordfast[1])**2)).astype(int)
      tb = (30 + 2*np.sqrt((x-coordbulk[0])**2 + (y-coordbulk[1])**2)).astype(int)
      ta = (30 + 2*np.sqrt((x-coordasrs[0])**2 + (y-coordasrs[1])**2)).astype(int)
      tx = (30 + 2*np.sqrt((x-coordxdock[0])**2 + (y-coordxdock[1])**2)).astype(int)
      df.loc[len(df)] = [tf, tb, ta, tx]
   return df

def is_valid(matrix):
    """Check if matrix meets all constraints."""
    # All non-negative integers
    if not np.all(matrix >= 0): return False
    if not np.all(matrix.astype(int) == matrix): return False

    # Column sums fixed
    if not np.all(matrix.sum(axis=0) == matrix[0].sum(axis=0)): return False

    # Row sums in bounds
    row_sums = matrix.sum(axis=1)
    if not np.all((row_sums >= 16) & (row_sums <= 28)): return False

    # At most one zero per row
    if not np.all((matrix == 0).sum(axis=1) <= 1): return False

    # Monotonically decreasing per row
    if not np.all(np.diff(matrix, axis=1) <= 0): return False

    return True

def perturb_matrix(matrix, n_attempts=1000):
    """Randomly perturb matrix while preserving all constraints."""
    matrix = matrix.copy()
    n_rows, n_cols = matrix.shape
    col_sums = matrix.sum(axis=0)

    for _ in range(n_attempts):
        i = np.random.randint(n_rows)
        j = np.random.randint(3)  # Only up to column 2 to swap with j+1

        # Decide how much to transfer between x[i][j] and x[i][j+1]
        max_delta = min(matrix[i][j] - matrix[i][j+1], matrix[i][j] // 2)
        if max_delta <= 0:
            continue

        delta = np.random.randint(1, max_delta + 1)

        # Try decreasing j, increasing j+1 (still keeping monotonicity)
        trial = matrix.copy()
        trial[i][j] -= delta
        trial[i][j+1] += delta

        if is_valid(trial):
            return trial  # Return on first valid perturbation

    return matrix  # No valid perturbation found

def generate_matrix_pulp(nfast, nbulk, nasrs, nxdock, ntruck):
   ncols = 4
   col_targets = [nfast, nbulk, nasrs, nxdock]
   M = max(col_targets)  # for big-M logic
   
   # Problem definition
   prob = pulp.LpProblem("Truck_Load_Assignment", pulp.LpMaximize)
   
   # Integer variables: x[i][j]
   x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Integer') for j in range(ncols)] for i in range(ntruck)]
   
   # Binary indicators: is_zero[i][j]
   is_zero = [[pulp.LpVariable(f"is_zero_{i}_{j}", cat='Binary') for j in range(ncols)] for i in range(ntruck)]
   
   # Random objective to increase diversity
   weights = np.random.rand(ntruck, ncols)
   prob += pulp.lpSum(weights[i][j] * x[i][j] for i in range(ntruck) for j in range(ncols))
   
   # Column sum constraints
   for j in range(ncols):
      prob += pulp.lpSum(x[i][j] for i in range(ntruck)) == col_targets[j]
   
   # Row sum constraints
   for i in range(ntruck):
      row_sum = pulp.lpSum(x[i][j] for j in range(ncols))
      prob += row_sum >= 18
      prob += row_sum <= 26
   
   # Zero detection and at most one zero per row
   for i in range(ntruck):
      for j in range(ncols):
         prob += x[i][j] <= M * (1 - is_zero[i][j])  # is_zero == 1 ⇒ x == 0
         prob += x[i][j] >= 1 - M * is_zero[i][j]  # is_zero == 0 ⇒ x ≥ 1
      prob += pulp.lpSum(is_zero[i][j] for j in range(ncols)) <= 1
      
      # Monotonic decreasing constraints: col 0 ≥ col 1 ≥ col 2 ≥ col 3
      prob += x[i][1] <= x[i][0] -5
      prob += x[i][2] <= x[i][1] -1
      prob += x[i][3] <= x[i][2]
   
   # Solve
   result = prob.solve()
   if pulp.LpStatus[result] != 'Optimal':
      raise ValueError("No feasible solution found.")
   
   matrix = np.array([[int(pulp.value(x[i][j])) for j in range(ncols)] for i in range(ntruck)])
   return matrix

if __name__ == '__main__':
   np.random.seed(995)
   dfpred = pd.read_csv('../boost_forecast/res_YW_AR_75.csv')
   preds = dfpred['true'].values

   dfDist = computeDistances()

   for iday in range(0,1):
      nfast = preds[iday*4+0]
      nbulk = preds[iday*4+1]
      nasrs = preds[iday*4+2]
      nxdock = preds[iday*4+3]

      avgLoad = 22
      ntruck = round(np.sum(preds[iday*4:iday*4+4])/avgLoad)
      result = generate_matrix_pulp(nfast, nbulk, nasrs, nxdock, ntruck)
      print(result)
      print(f"Target column sums: {nfast}, {nbulk}, {nasrs}, {nxdock}")
      print("Column Sums:", result.sum(axis=0))
      print("Row Sums:", [r.sum() for r in result])
      
      for _ in range(10):
         result = perturb_matrix(result)
      print("Column Sums:", result.sum(axis=0))
      print("Row Sums:", [r.sum() for r in result])
      dfLoads = pd.DataFrame(result, columns=['tf', 'tb', 'ta', 'tx'])

   print('Finito')

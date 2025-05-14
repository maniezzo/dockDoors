import math

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose
import copy,random

def read_data(changepoint):
   df = pd.read_csv("../warehouse3.csv")
   # ds = df['MA1'].copy()  # converts to series
   # ds[ds==0] = 1
   # result = seasonal_decompose(ds[:300], model='multiplicative', period=12)
   # result.plot()
   # plt.show()
   # result = seasonal_decompose(ds[300:], model='multiplicative', period=12)
   # result.plot()
   # plt.show()

   daySeries = []
   offsets = [2, 3, 4, 5, 6, 0, 1]  # days from monday
   offinv  = [5, 6, 0, 1, 2, 3, 4]  # days back from sat

   for offset in offsets:
      indices = range(offset, len(df), 7)  # Every 7th row starting at offset
      series = df.loc[indices, 'LowFri'].reset_index(drop=True)  # Optional: reset index
      daySeries.append(pd.DataFrame(series))

   df['newvals'] = 0
   for i in range(len(daySeries)):
      # Split the data
      first_segment  = daySeries[i]['LowFri'].iloc[:changepoint]
      second_segment = daySeries[i]['LowFri'].iloc[changepoint:]
      x1 = np.arange(len(first_segment))
      x2 = np.arange(len(first_segment), len(daySeries[i]['LowFri']))

      # regressions
      slope1, intercept1, r1, _, _ = linregress(x1, first_segment)
      slope2, intercept2, r2, _, _ = linregress(x2, second_segment)

      # Compute fitted values
      regression1 = slope1 * x1 + intercept1
      regression2 = slope2 * x2 + intercept2

      # Add regression column
      daySeries[i]['Regression'] = np.nan
      daySeries[i].loc[:changepoint-1, 'Regression'] = regression1
      daySeries[i].loc[changepoint:, 'Regression']   = regression2
      residuals2 = second_segment - regression2
      third_segment = second_segment.copy()
      if(i<6):
         ii = 0
         for idx in third_segment.index:
            slider = 3.0*(ii/len(third_segment)-0.5)
            if(abs(residuals2[idx]) > 1.2*residuals2.std()):
               third_segment[idx] = math.trunc(regression2[ii] + random.random()*slider*residuals2[idx])
            ii += 1

      showPlots = False
      if showPlots:
         plt.plot(daySeries[i]['LowFri'][:changepoint], marker='.',linewidth=0)
         plt.plot(x1, regression1, color='blue', label=f'Regression 1 ')
         plt.plot(x2, regression2, color='green', label=f'Regression 2')
         plt.plot(x2, third_segment, color='red', label=f'third_segment', marker='.',linewidth=0)
         plt.title(f"day {i}")
         plt.ylim([0,5000])
         plt.legend(loc='best')
         plt.show()

      # Combine ds1 and ds2
      combined_series = pd.concat([first_segment, third_segment])
      daySeries[i]["newvals"] = combined_series
      for j in range(len(combined_series)):
         jj = offsets[i] + 7*j
         df.loc[jj,'newvals'] = combined_series[j]

   df["newvals"] += df['production']

   plt.plot(df['pallets'], marker='.',linewidth=0)
   plt.plot(df['newvals'], marker='.',linewidth=0)
   plt.title("newvals")
   plt.show()

   df['newvals'].to_csv("../data/warehouse3new.csv", index=False)
   return daySeries

# calcola avg e std di ogni serie e definisce i punti in accordo
def resetNormal(s):
   avg = np.average(s)
   std = np.std(s)
   s1  = [random.gauss(avg, std) for _ in range(len(s))]
   return np.array(s1)

# definisce le aree di magazzino origine delle missioni
def splitAreas(changepoint,daySeries):
   # eree: fast, bulk, asrs, xdock
   percFast  = 0.5
   percBulk  = 0.15
   percAsrs  = 0.3
   percXdock = 0.05
   sumProb   = percFast + percBulk + percAsrs + percXdock
   assert abs(sumProb-1.0) < 0.0001, "probs must sum to 1"
   minlength = min(len(series) for series in daySeries)-changepoint

   df = pd.DataFrame()
   for i in range(len(daySeries)):
      dPercFast, dPercBulk, dPercAsrs, dPercXdock = percFast, percBulk, percAsrs, percXdock
      series = daySeries[i]['newvals'][changepoint:].values
      regr   = daySeries[i]['Regression'][changepoint:].values
      fast  = np.zeros(len(series))
      bulk  = np.zeros(len(series))
      asrs  = np.zeros(len(series))
      xdock = np.zeros(len(series))
      if (i==5 or i==6):
         dPercFast  += dPercAsrs
         dPercBulk  += dPercXdock
         dPercAsrs  = 0
         dPercXdock = 0

      for j in range(len(series)):
         fast[j]  = series[j]*dPercFast
         bulk[j]  = series[j]*dPercBulk*(1-0.4*random.random())
         asrs[j]  = regr[j]*dPercAsrs*(1-0.3*random.random())
         xdock[j] = regr[j]*dPercXdock*(1-0.5*random.random())
         slack = series[j] - fast[j] - bulk[j] - asrs[j] - xdock[j]
         fast[j] = fast[j] + slack
         if(fast[j] < 0): fast[j] = random.randrange(5,10)

      fastNormal  = resetNormal(fast[:minlength])
      bulkNormal  = resetNormal(bulk[:minlength])
      asrsNormal  = resetNormal(asrs[:minlength])
      xdockNormal = resetNormal(xdock[:minlength])

      df[f"fast{i}"]  = fastNormal.astype(int)
      df[f"bulk{i}"]  = bulkNormal.astype(int)
      df[f"asrs{i}"]  = asrsNormal.astype(int)
      df[f"xdock{i}"] = xdockNormal.astype(int)

   df.to_csv("../data/flows.csv", index=False)
   return

if __name__ == '__main__':
   random.seed(995)
   changepoint = 50
   daySeries   = read_data(changepoint)

   splitAreas(changepoint,daySeries)

   print("fine")


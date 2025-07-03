# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # for heatmap
import os
import operator # lt, ge and the like
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx
import scikit_posthocs as sp

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
   """
   Draws a CD graph, which is used to display  the differences in methods'
   performance. See Janez Demsar, Statistical Comparisons of algorithms over
   Multiple Data Sets, 7(Jan):1--30, 2006.

   Needs matplotlib to work.

   The image is ploted on `plt` imported using
   `import matplotlib.pyplot as plt`.

   Args:
       avranks (list of float): average ranks of methods.
       names (list of str): names of methods.
       cd (float): Critical difference used for statistically significance of
           difference between methods.
       cdmethod (int, optional): the method that is compared with other methods
           If omitted, show pairwise comparison of methods
       lowv (int, optional): the lowest shown rank
       highv (int, optional): the highest shown rank
       width (int, optional): default width in inches (default: 6)
       textspace (int, optional): space on figure sides (in inches) for the
           method names (default: 1)
       reverse (bool, optional):  if set to `True`, the lowest rank is on the
           right (default: `False`)
       filename (str, optional): output file name (with extension). If not
           given, the function does not write a file.
       labels (bool, optional): if set to `True`, the calculated avg rank
       values will be displayed
   """
   
   width = float(width)
   textspace = float(textspace)
   
   def nth(l, n):
      #Returns the nth elemnt in a list.
      n = lloc(l, n)
      return [a[n] for a in l]
   
   def lloc(l, n):
      """
      List location in list of list structure.
      Enable the use of negative locations:
      -1 is the last element, -2 second last...
      """
      if n < 0:
         return len(l[0]) + n
      else:
         return n
   
   def mxrange(lr):
      """
      Multiple xranges. Can be used to traverse matrices.
      This function is very slow due to unknown number of
      parameters.

      >>> mxrange([3,5])
      [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

      >>> mxrange([[3,5,1],[9,0,-3]])
      [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

      """
      if not len(lr):
         yield ()
      else:
         # it can work with single numbers
         index = lr[0]
         if isinstance(index, int):
            index = [index]
         for a in range(*index):
            for b in mxrange(lr[1:]):
               yield tuple([a] + list(b))
   
   sums   = avranks
   nnames = names
   ssums  = sums
   
   if lowv is None:
      lowv = min(1, int(math.floor(min(ssums))))
   if highv is None:
      highv = max(len(avranks), int(math.ceil(max(ssums))))
   
   cline = 0.4 # affects distance btween title and plot
   
   k = len(sums)
   lines = None
   linesblank = 0
   scalewidth = width - 2 * textspace
   
   def rankpos(rank):
      if not reverse:
         a = rank - lowv
      else:
         a = highv - rank
      return textspace + scalewidth / (highv - lowv) * a
   
   distanceh = 0.4 # between title and plot
   
   cline += distanceh
   
   # calculate needed height of the image
   minnotsignificant = max(2 * 0.2, linesblank)
   height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant
   
   fig = plt.figure(figsize=(width, height))
   fig.set_facecolor('white')
   ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
   ax.set_axis_off()
   
   hf = 1. / height  # height factor
   wf = 1. / width
   
   def hfl(l):
      return [a * hf for a in l]
   
   def wfl(l):
      return [a * wf for a in l]
   
   # Upper left corner is (0,0).
   ax.plot([0, 1], [0, 1], c="w")
   ax.set_xlim(0, 1)
   ax.set_ylim(1, 0)
   
   def line(l, color='k', **kwargs):
      """
      Input is a list of pairs of points.
      """
      ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)
   
   def text(x, y, s, *args, **kwargs):
      ax.text(wf * x, hf * y, s, *args, **kwargs)
   
   line([(textspace, cline), (width - textspace, cline)], linewidth=2)
   
   bigtick = 0.3
   smalltick = 0.15
   linewidth = 2.0
   linewidth_sign = 4.0
   
   tick = None
   for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
      tick = smalltick
      if a == int(a):
         tick = bigtick
      line([(rankpos(a), cline - tick / 2),
            (rankpos(a), cline)],
           linewidth=2)
   
   for a in range(lowv, highv + 1):
      text(rankpos(a), cline - tick / 2 - 0.05, str(a),
           ha="center", va="bottom", size=16)
   
   k = len(ssums)
   
   def filter_names(name):
      return name
   
   space_between_names = 0.24
   
   for i in range(math.ceil(k / 2)):
      chei = cline + minnotsignificant + i * space_between_names
      line([(rankpos(ssums[i]), cline),
            (rankpos(ssums[i]), chei),
            (textspace - 0.1, chei)],
           linewidth=linewidth)
      if labels:
         text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
      text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)
   
   for i in range(math.ceil(k / 2), k):
      chei = cline + minnotsignificant + (k - i - 1) * space_between_names
      line([(rankpos(ssums[i]), cline),
            (rankpos(ssums[i]), chei),
            (textspace + scalewidth + 0.1, chei)],
           linewidth=linewidth)
      if labels:
         text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
      text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]), ha="left", va="center", size=16)
   
   # no-significance lines
   def draw_lines(lines, side=0.05, height=0.1):
      start = cline + 0.2
      
      for l, r in lines:
         line([(rankpos(ssums[l]) - side, start),
               (rankpos(ssums[r]) + side, start)],
              linewidth=linewidth_sign)
         start += height
         print('drawing: ', l, r)
   
   # draw_lines(lines)
   start = cline + 0.2
   side = -0.02
   height = 0.1
   
   # draw no significant lines
   # get the cliques
   cliques = form_cliques(p_values, nnames)
   i = 1
   achieved_half = False
   print(nnames)
   for clq in cliques:
      if len(clq) == 1:
         continue
      print(clq)
      min_idx = np.array(clq).min()
      max_idx = np.array(clq).max()
      if min_idx >= len(nnames) / 2 and achieved_half == False:
         start = cline + 0.25
         achieved_half = True
      line([(rankpos(ssums[min_idx]) - side, start),
            (rankpos(ssums[max_idx]) + side, start)],
           linewidth=linewidth_sign)
      start += height

def form_cliques(p_values, nnames):
   # This method forms the cliques, first form the numpy matrix data
   m = len(nnames)
   g_data = np.zeros((m, m), dtype=np.int64)
   for p in p_values:
      if p[3] == False:
         i = np.where(nnames == p[0])[0][0]
         j = np.where(nnames == p[1])[0][0]
         min_i = min(i, j)
         max_j = max(i, j)
         g_data[min_i, max_j] = 1
   
   g = networkx.Graph(g_data)
   return networkx.find_cliques(g)

def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, fAscending = False):
   # Draws the critical difference diagram given the list of pairwise algorithms that are significant or not
   p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha, fAscending = fAscending)
   
   print(f"average_ranks {average_ranks}")
   
   with open("ranks_results.txt", "a") as file:
      # Write the text you want to append
      file.write(f"{title} ranks {average_ranks.index.values}\n")
      file.write(f"{title} ranks {average_ranks.values}\n")
   
   for p in p_values:
      print(f"p values: {p}")
   
   graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
               cd=None, reverse=not fAscending, width=9, textspace=1.5, labels=labels)
   
   font = {'family': 'sans-serif',
           'color':  'black',
           'weight': 'normal',
           'size': 18,
           }
   if title:
      plt.title(title+" ranks",fontdict=font, y=0.85, x=0.5)
   plt.savefig(f'{title}.eps',bbox_inches='tight')
   plt.show()
   
   # here for the scikit version
   names = df_perf[algonames].unique().tolist()
   for name in names:
      p_values.append((name,name,1))
   # Create the p_values DataFrame
   dfpval = pd.DataFrame(p_values, columns=['index', 'columns', 'value', "dummy"])
   # Pivot the DataFrame to get the desired shape
   dfpval = dfpval.pivot(index='index', columns='columns', values='value')
   # Copy the upper triangular value to the lower triangular position
   for i in range(len(dfpval)):
      for j in range(i):
         dfpval.iat[i, j] = dfpval.iat[j, i]
   # Set the index and column names (already correctly set by pivot)
   dfpval.index.name = None
   dfpval.columns.name = None
   
   # Create a heatmap with numerical values
   sns.set(style='white')  # Optional: set the seaborn style
   plt.figure(figsize=(6, 4))  # Optional: set the size of the figure
   
   ax = sns.heatmap(dfpval, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
   plt.title('Heatmap with Numerical Values')
   plt.xlabel('X-axis Label')
   plt.ylabel('Y-axis Label')
   plt.show()
   
   # scikit posthocs critical difference diagram
   plt.figure(figsize=(10, 3), dpi=128)
   plt.title(f'sp CDD of avg ranks for {title}, data')
   sp.critical_difference_diagram(average_ranks, dfpval,
                                  text_h_margin=0.8,
                                  crossbar_props={'color': None, 'marker': '.'})
   plt.subplots_adjust(top=0.8)
   plt.savefig(f'sp {title}.eps',bbox_inches='tight')
   plt.show()
   return

def wilcoxon_holm(alpha=0.05, df_perf=None, fAscending = False):
   """
   Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
   to reject the null's hypothesis
   """
   print(pd.unique(df_perf[algonames]))
   # count the number of tested datasets per algorithm
   df_counts = pd.DataFrame({'count': df_perf.groupby(
      [algonames]).size()}).reset_index()
   # get the maximum number of tested datasets
   max_nb_datasets = df_counts['count'].max()
   # get the list of algorithms who have been tested on nb_max_datasets
   algorithms = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                     [algonames])
   # test the null hypothesis using friedman before doing a post-hoc analysis
   friedman_p_value = friedmanchisquare(*(
      np.array(df_perf.loc[df_perf[algonames] == c][obfuncname])
      for c in algorithms))[1]
   if friedman_p_value >= alpha:
      # then the null hypothesis over the entire algorithms cannot be rejected
      print('the null hypothesis over the entire algorithm set cannot be rejected')
      exit()
   # get the number of algorithms
   m = len(algorithms)
   # init array that contains the p-values calculated by the Wilcoxon signed rank test
   p_values = []
   # loop through the algorithms to compare pairwise
   for i in range(m - 1):
      # get the name of algorithm one
      algorithm_1 = algorithms[i]
      # get the performance of algorithm one
      perf_1 = np.array(df_perf.loc[df_perf[algonames] == algorithm_1][obfuncname]
                        , dtype=np.float64)
      for j in range(i + 1, m):
         # get the name of the second algorithm
         algorithm_2 = algorithms[j]
         # get the performance of algorithm one
         perf_2 = np.array(df_perf.loc[df_perf[algonames] == algorithm_2]
                           [obfuncname], dtype=np.float64)
         # calculate the p_value
         p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
         # appen to the list
         p_values.append((algorithm_1, algorithm_2, p_value, False))
   # get the number of hypothesis
   k = len(p_values)
   # sort the list in acsending manner of p-value
   p_values.sort(key=operator.itemgetter(2))
   
   # loop through the hypothesis
   for i in range(k):
      # correct alpha with holm
      new_alpha = float(alpha / (k - i))
      # test if significant after holm's correction of alpha
      if p_values[i][2] <= new_alpha:
         p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
      else:
         # stop
         break
   # compute the average ranks to be returned (useful for drawing the cd diagram)
   # sort the dataframe of performances
   sorted_df_perf = df_perf.loc[df_perf[algonames].isin(algorithms)]. \
      sort_values([algonames, instancenames])
   # get the rank data
   rank_data = np.array(sorted_df_perf[obfuncname]).reshape(m, max_nb_datasets)
   
   # create the data frame containg the accuracies
   df_ranks = pd.DataFrame(data=rank_data, index=np.sort(algorithms), columns=
   np.unique(sorted_df_perf[instancenames]))
   
   # number of wins
   dfff = df_ranks.rank(ascending=fAscending)
   print(dfff[dfff == 1.0].sum(axis=1))
   
   # average the ranks
   average_ranks = df_ranks.rank(ascending=fAscending).mean(axis=1).sort_values(ascending=fAscending)
   # return the p-values and the average ranks
   return p_values, average_ranks, max_nb_datasets

def draw_diagram(df_perf, dfname=""):
   print("Wilcoxon signed-rank test with Bonferroni-Holm correction")
   global algonames
   algonames     = df_perf.columns[0]
   global instancenames
   instancenames = df_perf.columns[1]
   global obfuncname
   obfuncname    = df_perf.columns[2]
   # fAscending True:  minimize, lower values better
   # fAscending False: maximize, higher values better
   draw_cd_diagram(df_perf=df_perf, title=f"{obfuncname} {dfname}", labels=True, fAscending = True)
   print(f"Diagram is written on file {obfuncname}resXXX.png")

if __name__ == '__main__':
   os.chdir(os.path.dirname(os.path.abspath(__file__)))
   print("Wilcoxon signed-rank test with Bonferroni-Holm correction")
   df_perf = pd.read_csv('results.csv',index_col=False)
   algonames     = df_perf.columns[0]
   instancenames = df_perf.columns[1]
   obfuncname    = df_perf.columns[2]
   # fAscending True:  minimize, lower values better
   # fAscending False: maximize, higher values better
   draw_cd_diagram(df_perf=df_perf, title=obfuncname, labels=True, fAscending = True)
   print(f"Diagram is written on file {obfuncname}resXXX.eps")



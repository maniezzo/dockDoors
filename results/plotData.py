import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plot_series(ds,day):
   fig,ax = plt.subplots(figsize=(6,6))
   plt.plot(ds)
   ax.set_ylim(0, 5500)  # lower, upper
   plt.xlabel("Week Day Index",size=20)
   plt.title(f"{day} avg:{ds.mean():.0f} std:{ds.std():.0f}",size=20)
   ax.tick_params(labelsize=16)
   ax.grid(visible=True,which='both', axis='both', alpha = 0.3)
   plt.tight_layout()
   plt.savefig(f"{day}.png")
   plt.show()
   return

if __name__ == "__main__":
   df = pd.read_excel("../warehouse3new.xlsx")
   ds = df.loc[:,"sab"].dropna()
   plot_series(ds,"Saturday")
   ds = df.loc[:,"dom"].dropna()
   plot_series(ds,"Sunday")
   ds = df.loc[:,"lun"].dropna()
   plot_series(ds,"Monday")
   ds = df.loc[:,"mar"].dropna()
   plot_series(ds,"Tuesday")
   ds = df.loc[:,"mer"].dropna()
   plot_series(ds,"Wednesday")
   ds = df.loc[:,"gio"].dropna()
   plot_series(ds,"Thursday")
   ds = df.loc[:,"ven"].dropna()
   plot_series(ds,"Friday")
   print("fine")
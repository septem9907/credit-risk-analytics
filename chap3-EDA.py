# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
mortgage = pd.read_csv("./data/mortgage.csv")

# %%
mortgage.info()

# %%
# Observed Frequencies and Empirical Distributions

plt.figure()
plt.hist(mortgage.FICO_orig_time, bins=100, density=True)
plt.xlabel("FICO_orig_time")
plt.ylabel("Density")
plt.show()

# %%
plt.figure()
plt.hist(mortgage.FICO_orig_time, bins=100, density=True, cumulative=True)
plt.xlabel("FICO_orig_time")
plt.ylabel("Cumulative Percent")
plt.show()

# %%

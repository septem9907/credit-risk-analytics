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
result = (
    mortgage.groupby(["default_time"], as_index=False)
        .id.count()
        .rename(columns={"id": "frequency"})
        .assign(percent=lambda x: x.frequency / x.frequency.sum() * 100)
)
print(result)

# %%
# Location Measure

# %%
loc_measures = (
    mortgage.loc[:, ["default_time", "FICO_orig_time", "LTV_orig_time"]]
        .describe()
        .transpose()
)
print(loc_measures)

# %%
from statsmodels.graphics.gofplots import qqplot
qqplot(mortgage.FICO_orig_time)

# %%
import seaborn as sns
sns.set(style="whitegrid")

# %%
sns.boxplot(x="default_time", y="FICO_orig_time", data=mortgage)

# %%
(
    mortgage.groupby(["default_time"])
        .boxplot(column=["FICO_orig_time"])
)

# %%

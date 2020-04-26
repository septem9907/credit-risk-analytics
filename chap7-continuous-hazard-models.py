# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
mortgage = pd.read_csv("./data/mortgage.csv")
mortgage.head()

# %%

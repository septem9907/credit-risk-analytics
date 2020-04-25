# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
mortgage = pd.read_csv("./data/mortgage.csv")
mortgage.head()

# %%
# 1. Linear Model

# %%
linear_model = smf.ols("default_time ~ FICO_orig_time + LTV_orig_time + gdp_time", data=mortgage).fit()
linear_model.summary()

# %%
table = sm.stats.anova_lm(linear_model, typ=1)
print(table)

# %%
# 2. Nonlinear Models

# %%
import matplotlib.pyplot as plt
import scipy.stats as stats

x = np.arange(-10, 10, step=0.1)

plt.plot(x, stats.norm.cdf(x, 0, 1), color="blue", linestyle="dashed", linewidth=2, markersize=12)
plt.plot(x, np.exp(x)/(1+np.exp(x)))
plt.plot(x, 1-np.exp(-np.exp(x)))
plt.xlabel("Linear predictor")
plt.ylabel("Probability")
plt.legend(["probit", "logit", "cloglog"])
plt.show()

# %%
mortgage2 = mortgage.loc[mortgage.notnull().all(axis=1), :]
logit_model = smf.glm("default_time ~ FICO_orig_time + LTV_time + gdp_time",
    data=mortgage,
    family=sm.families.Binomial(link=sm.families.links.logit)).fit()
logit_model.summary()

# %%
logit_model.params
logit_model.normalized_cov_params

# %%
logit_model.wald_test_terms()

# %%

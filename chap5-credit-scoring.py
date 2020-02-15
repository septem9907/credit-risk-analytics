# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
hmeq = pd.read_csv("./data/hmeq.csv")


# %%
flag = (
    hmeq.drop(columns=["JOB", "REASON"])
        .notnull()
        .all(axis=1)
)

hmeq_omit = hmeq.loc[flag, :]

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
hmeq_final = smf.glm(formula="BAD ~ DEBTINC + DELINQ + DEROG + CLAGE + NINQ + CLNO + JOB",
    data=hmeq_omit,
    family=sm.families.Binomial(link=sm.families.links.logit)).fit()
hmeq_final.summary()

# %%
from sklearn.linear_model import LogisticRegression

# %%
col_X = ["DEBTINC", "DELINQ", "DEROG", "CLAGE", "NINQ", "CLNO", "JOB"]
col_y = ["BAD"]
X = hmeq_omit.loc[:, col_X]
y = hmeq_omit.loc[:, col_y]

# %%
X = (
    pd.concat([X.drop(columns=["JOB"]), pd.get_dummies(X["JOB"], prefix="JOB")], axis=1)
)

# %%
clf = LogisticRegression().fit(X, y)

# %%
clf.intercept_
clf.coef_

# %%
clf.score(X, y)

# %%
from sklearn.metrics import roc_curve, roc_auc_score
pred = clf.predict(X)

fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
auc_train = roc_auc_score(y, pred)

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc_train)
plt.plot([0, 1], [0, 1], color="darkorange", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# %%

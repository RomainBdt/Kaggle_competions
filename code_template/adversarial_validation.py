# %% Adversarial Validation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pandas as pd

df1 = pd.read_csv('datasets/train.csv')
df2 = pd.read_csv('datasets/winequality-red.csv')

df1['data'] = 0
df2['data'] = 1
df1.drop(['Id', 'quality'], axis=1, inplace=True)
df2.drop(['quality'], axis=1, inplace=True)
    
X = pd.concat([df1, df2], ignore_index=True)
y = X.pop('data')

model = lgb.LGBMClassifier()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(roc_auc_score(y_te, y_pred))

scores = cross_val_score(lgb.LGBMClassifier(), X, y, cv=10, scoring='roc_auc')
print(scores)

# %%
# show features importance
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

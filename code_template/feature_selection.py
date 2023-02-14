# %% Drop one feature at a time and evaluate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMRegressor

def scorer(estimator, X, original_labels):
    y_pred = estimator.predict(X)
    return cohen_kappa_score(original_labels, y_pred, weights='quadratic')

estimator = LGBMRegressor(random_state=42, 
                        #   n_jobs=-1, 
                          boosting_type='rf', 
                          subsample=.632, 
                          subsample_freq=1)

# drop one feature at a time and evaluate
cols = X_train.columns
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
col_score = {}
col_score_val = {}
remaining_cols = {}
len_cols = len(cols)
for i in reversed(range(1, len_cols)):
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select = len(cols) - 1,
        direction = 'backward',
        scoring=scorer,
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X_train[cols], y_train)
    cols = X_train[cols].columns[sfs.get_support()]
    
    col_score[i] = cross_val_score(estimator, X_train[cols], y_train, cv=cv, scoring=scorer).mean()
    estimator.fit(X_train[cols], y_train)
    col_score_val[i] = scorer(estimator, X_val[cols], y_val)
    
    remaining_cols[i] = cols
    print(f'{i} features: {col_score[i]:.3f} {col_score_val[i]:.3f} {len(cols)}')

# %%
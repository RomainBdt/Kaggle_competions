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
# from sklearn.feature_selection import SequentialFeatureSelector

# # drop rows with inf values for SFS
# for df in [X_train, X_test]:
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.dropna(inplace=True)

# # align target
# y_train = y_train.loc[X_train.index]
# y_test = y_test.loc[X_test.index]

# cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# # drop one feature at a time and evaluate
# cols = X_train.columns
# selected_cols = []
# col_score = {}
# len_cols = len(cols)

# regressor = LGBMRegressor(random_state=SEED, n_jobs=-1)
# estimator = TransformedTargetRegressor(regressor=regressor, func=target_transform, inverse_func=inverse_target_transform, check_inverse=False)

# for nb_cols in range(1, len_cols):
#     print(f'Number of features: {nb_cols}')
#     best_score = np.inf
#     for col in tqdm(cols):
#         if col in selected_cols:
#             continue
#         score = -cross_val_score(estimator, X_train[selected_cols + [col]], y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
#         if score < best_score:
#             best_score = score
#             best_col = col
#         # print(f'Features: {selected_cols + [col]}, score: {score}')
#     selected_cols.append(best_col)
#     estimator.fit(X_train[selected_cols], y_train)
#     y_pred = estimator.predict(X_test[selected_cols])
#     print(f'New column: {best_col}, CV score: {best_score}, test score: {(mean_squared_error(y_test, y_pred, squared=False))}', end='\n\n')
#     col_score[nb_cols] = best_score
    
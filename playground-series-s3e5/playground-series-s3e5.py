# %%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from functools import partial
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# import regressors
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, PassiveAggressiveRegressor, Perceptron, RidgeClassifier, LogisticRegression
from sklearn.linear_model import Lasso, ElasticNet, Lars, BayesianRidge, ARDRegression, OrthogonalMatchingPursuit, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.linear_model import LassoLars, LassoLarsIC
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# pandas deactivate future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
train = pd.read_csv('datasets/train.csv')
train.drop('Id', axis=1, inplace=True)
orig_train = pd.read_csv('datasets/WineQT.csv')
orig_train.drop('Id', axis=1, inplace=True)
X_test = pd.read_csv('datasets/test.csv')
X_test.drop('Id', axis=1, inplace=True)
len_test = X_test.shape[0]
red_wine = pd.read_csv('datasets/winequality-red.csv')

RUN_FOR_FINAL_PREDICTION = True

# Split data
X_train, X_val = train_test_split(train, test_size=0.2, random_state=42, stratify=train['quality'])
y_val = X_val.pop('quality')

# Add origin data
X_train = pd.concat([X_train, orig_train, red_wine]).reset_index(drop=True)
X_train.drop_duplicates(inplace=True)

# # Outlier removal
# print(X_train.shape)
# X_train = X_train[X_train['volatile acidity'] <= 1.33]
# X_train = X_train[X_train['citric acid'] <= 0.8]
# X_train = X_train[X_train['residual sugar'] < 9]
# X_train = X_train[X_train['chlorides'] <= 0.5]
# X_train = X_train[X_train['total sulfur dioxide'] <= 200]
# X_train = X_train[X_train['sulphates'] < 1.9]
# X_train = X_train[X_train['alcohol'] <= 14]
# print(X_train.shape)

y_train = X_train.pop('quality')

# Data preprocessing
def add_features(df):
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383685
    df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
    df['free_sulfur/total_sulfur'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    df['sugar/alcohol'] = df['residual sugar'] / df['alcohol']
    df['alcohol/density'] = df['alcohol'] / df['density']
    df['total_acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['sulphates/chlorides'] = df['sulphates'] / df['chlorides']
    df['bound_sulfur'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
    df['alcohol/pH'] = df['alcohol'] / df['pH']
    df['alcohol/acidity'] = df['alcohol'] / df['total_acid']
    df['alkalinity'] = df['pH'] + df['alcohol']
    df['mineral'] = df['chlorides'] + df['sulphates'] + df['residual sugar']
    df['density/pH'] = df['density'] / df['pH']
    df['total_alcohol'] = df['alcohol'] + df['residual sugar']
    
    # From https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698
    df['acid/density'] = df['total_acid']  / df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']
    df['sulphates/acid'] = df['sulphates'] / df['volatile acidity']
    df['sulphates*alcohol'] = df['sulphates'] * df['alcohol']
    
    return df

for df in [X_train, X_val, X_test]:
    df = add_features(df)
    
# Label encode
min_y = min(y_train.min(), y_val.min())
y_train -= min_y
y_val -= min_y

if RUN_FOR_FINAL_PREDICTION:
    X_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_train = pd.concat([y_train, y_val]).reset_index(drop=True)
    

# %% Regressors

regressors = {
    # 'LGBMRegressor1': LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='gbdt'),
    # 'LGBMRegressor2': LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='dart'),
    # 'LGBMRegressor3': LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='goss'),
    'LGBMRegressor4': LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='rf', subsample=.632, subsample_freq=1),
    # 'LGBMRegressor5': LGBMRegressor(random_state=42, n_jobs=-1, class_weight='balanced'),
    # 'LGBMRegressor6': LGBMRegressor(random_state=42, n_jobs=-1, subsample=0.7),
    # 'LGBMRegressor7': LGBMRegressor(random_state=42, n_jobs=-1, colsample_bytree=0.7),
    # 'LGBMRegressor8': LGBMRegressor(random_state=42, n_jobs=-1, subsample=0.7, colsample_bytree=0.7),
    # 'LGBMRegressor9': LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='dart', colsample_bytree=0.7),
    # 'XGBRegressor1': XGBRegressor(random_state=42, n_jobs=-1),
    # 'XGBRegressor2': XGBRegressor(random_state=42, n_jobs=-1, booster='dart'),
    # 'XGBRegressor3': XGBRegressor(random_state=42, n_jobs=-1, booster='gblinear'),
    # 'XGBRegressor4': XGBRegressor(random_state=42, n_jobs=-1, colsample_bytree=0.7),
    # 'XGBRegressor5': XGBRegressor(random_state=42, n_jobs=-1, subsample=0.7),
    'XGBRandomForestRegressor': XGBRFRegressor(random_state=42, n_jobs=-1),
    # 'CatBoostRegressor': CatBoostRegressor(random_state=42, silent=True),
    'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    # 'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42, n_jobs=-1),
    # 'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    # 'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    # 'BaggingRegressor': BaggingRegressor(random_state=42, n_jobs=-1),
    # 'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1),
    # 'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    # 'GaussianProcessRegressor': GaussianProcessRegressor(random_state=42),
    # 'MLPRegressor1': MLPRegressor(random_state=42, max_iter=1000, activation='relu', solver='adam'),
    # 'MLPRegressor2': MLPRegressor(random_state=42, max_iter=1000, activation='relu', solver='lbfgs'),
    # 'MLPRegressor3': MLPRegressor(random_state=42, max_iter=5000, activation='tanh', solver='adam'),
    # 'MLPRegressor4': MLPRegressor(random_state=42, max_iter=1000, activation='tanh', solver='lbfgs'),
    # 'MLPRegressor5': MLPRegressor(random_state=42, max_iter=5000, activation='logistic', solver='adam'),
    # 'MLPRegressor6': MLPRegressor(random_state=42, max_iter=1000, activation='logistic', solver='lbfgs'),
    # 'MLPRegressor7': MLPRegressor(random_state=42, max_iter=5000, activation='identity', solver='adam'),
    # 'MLPRegressor8': MLPRegressor(random_state=42, max_iter=5000, activation='identity', solver='lbfgs'),
    # 'Ridge': Ridge(random_state=42),
    # 'SGDRegressor': SGDRegressor(random_state=42, max_iter=1000, tol=1e-3),
    # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=42, max_iter=1000, tol=1e-3),
    # 'Perceptron': Perceptron(random_state=42, max_iter=1000, tol=1e-3),
    # 'LinearRegression': LinearRegression(),
    # 'Lasso': Lasso(random_state=42),
    # 'ElasticNet': ElasticNet(random_state=42),
    # 'HuberRegressor': HuberRegressor(max_iter=1000),
    # 'BayesianRidge': BayesianRidge(),
    # 'ARDRegression': ARDRegression(),
    # 'TheilSenRegressor': TheilSenRegressor(random_state=42),
    # 'RANSACRegressor': RANSACRegressor(random_state=42),
    # 'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(normalize=False),
    # 'Lars': Lars(),
    # 'LassoLars': LassoLars(),
    # 'LassoLarsIC': LassoLarsIC(normalize=False),
    # 'StackingRegressor': StackingRegressor(
    #         estimators=[
    #             ('LGBMRandomForestRegressor', LGBMRegressor(random_state=42, n_jobs=-1, boosting_type='rf', subsample=.632, subsample_freq=1)),
    #             ('XGBRandomForestRegressor', XGBRFRegressor(random_state=42, n_jobs=-1)),
    #             ('RandomForestRegressor', RandomForestRegressor(random_state=42, n_jobs=-1)),
    #             # ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=42, n_jobs=-1))
    #             ], 
    #         final_estimator=Ridge(random_state=42),
    #         # cv=cv,
    #         # n_jobs=-1,
    #         )
}

# %% Regression with OptimizedRounder

class OptimizedRounder_v2(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5])
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5])
        return preds
    
    def coefficients(self):
        return self.coef_['x']
    
def scorer(estimator, X, original_labels):
    regression_predictions = estimator.predict(X)
    optR = OptimizedRounder_v2()
    optR.fit(regression_predictions, original_labels)
    y_pred = optR.predict(regression_predictions, optR.coefficients())
    return cohen_kappa_score(original_labels, y_pred, weights='quadratic')

def predict_score(estimator, optR, X, y_true=None):
    """Predict and round prediction with OptimizedRounder_v2
    return y_pred and cohen kappa score if y_true is present """
    
    regression_predictions = estimator.predict(X)
    y_pred = optR.predict(regression_predictions, optR.coefficients())
    
    if y_true is not None:
        score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    else: 
        score = None
    return y_pred, score


# %% Train models on subsets of data and make predictions
val_predictions = pd.DataFrame()
test_predictions = pd.DataFrame()
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('{: >30} {: >12} {: >12} {: >12} {: >12} {: >12} {: >12} {: >12}'.format(
    'Model', 'mean_cv_fold', 'std_cv_fold', 'min_cv_fold', 'mean_val', 'std_val', 'min_val', 'time'
    ))

for model_name, model in regressors.items():
    t0 = time.time()
    i = 0
    scores_cv = []
    scores_val = [] 
    for train_index, val_index in cv.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        y_val_fold = y_train.iloc[val_index]

        # fit model on training fold
        model.fit(X_train_fold, y_train_fold)
        
        # round predictions for test sets
        optR = OptimizedRounder_v2()
        optR.fit(model.predict(X_train_fold), y_train_fold)
        
        # eval on test fold
        y_val_fold_pred, score_val_fold = predict_score(model, optR, X_val_fold, y_val_fold)
        y_val_pred, score_val = predict_score(model, optR, X_val, y_val)
        y_test_pred, _ = predict_score(model, optR, X_test)
        
        scores_cv.append(score_val_fold)
        scores_val.append(score_val)
        val_predictions[model_name + str(i)] = y_val_pred
        test_predictions[model_name + str(i)] = y_test_pred
        i += 1
        
    row = ['%s' % model_name, 
        '%.3f' % np.mean(scores_cv), 
        '%.3f' % np.std(scores_cv),
        '%.3f' % (np.mean(scores_cv) - np.std(scores_cv)),
        '%.3f' % np.mean(scores_val), 
        '%.3f' % np.std(scores_val),
        '%.3f' % (np.mean(scores_val) - np.std(scores_val)),
        '%.3f' % (time.time() - t0),
        ]
    print('{: >30} {: >12} {: >12} {: >12} {: >12} {: >12} {: >12} {: >12}'.format(*row))
    results.append(np.array(scores_cv))

plt.figure(figsize=(25, 15))
plt.boxplot(results, labels=regressors.keys(), showmeans=True)
plt.show()

# %% Submission
y_pred_test = test_predictions.mean(axis=1).round().astype(int)
# remap predictions to original scale
y_pred_test += min_y

sub = pd.read_csv('submissions/sample_submission.csv')
sub['quality'] = y_pred_test
now = time.strftime("%Y-%m-%d %H_%M_%S")
sub.to_csv(f'submissions/submission{now}.csv', index=False)
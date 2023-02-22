# %%

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

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

SUBMIT = False
USE_ORIGINAL = False
SEED = 15
SAMPLE = 0.1

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
orig = pd.read_csv('datasets/cubic_zirconia.csv')

for i, df in enumerate([train, test, orig]):
    df.drop(['id'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    # df['dataset'] = i

# Define test set
if not SUBMIT:
    train, test = train_test_split(train, test_size=0.2, random_state=SEED) 

if USE_ORIGINAL:
    train = pd.concat([train, orig], axis=0)
    train.reset_index(inplace=True, drop=True)

# Sampling for faster training
if SAMPLE < 1:
    train = train.sample(frac=SAMPLE, random_state=SEED)

del orig

# transform categorical features
def transform_categorical(df):
    df['cut'] = df['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    df['color'] = df['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    df['clarity'] = df['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    return df

def remove_outliers(df):
    # Drop extreme values
    min = 2
    max = 20
    df = df[(df['x'] < max) & (df['y'] < max) & (df['z'] < max)]
    df = df[(df['x'] > min) & (df['y'] > min) & (df['z'] > min)]
    return df

def add_volume(df):
    df['volume'] = df['x'] * df['y'] * df['z']
    return df

def add_volume_ratio(df):
    df['volume_ratio1'] = (df['x'] * df['y']) / (df['z'] * df['z'])
    df['volume_ratio2'] = (df['x'] * df['z']) / (df['y'] * df['y'])
    df['volume_ratio3'] = (df['y'] * df['z']) / (df['x'] * df['x'])
    df['volume_ratio4'] = (df['x'] * df['y']) / (df['z'] * df['y'])
    df['volume_ratio5'] = (df['x'] * df['y']) / (df['z'] * df['x'])
    df['volume_ratio6'] = (df['x'] * df['z']) / (df['y'] * df['x'])
    df['volume_ratio7'] = (df['x'] * df['z']) / (df['y'] * df['z'])
    df['volume_ratio8'] = (df['y'] * df['z']) / (df['x'] * df['y'])
    df['volume_ratio9'] = (df['y'] * df['z']) / (df['x'] * df['z'])
    df['volume_ratio10'] = (df['x'] * df['y']) / (df['y'] * df['z'])
    df['volume_ratio11'] = (df['x'] * df['z']) / (df['x'] * df['y'])
    df['volume_ratio12'] = (df['x'] * df['y']) / (df['x'] * df['z'])
    return df

def add_surface_area(df):
    df['surface_area'] = 2 * (df['x'] * df['y'] + df['x'] * df['z'] + df['y'] * df['z'])
    return df

def transform_features(df):
    df = transform_categorical(df)
    # df = remove_outliers(df)
    # df = add_volume(df)
    # df = add_volume_ratio(df)
    # df = add_surface_area(df)
    return df

def target_transform(serie):
    # serie = serie.apply(lambda x: np.log1p(x))
    serie = np.log1p(serie)
    return serie

def inverse_target_transform(serie):
    # serie = serie.apply(lambda x: np.expm1(x))
    serie = np.expm1(serie)
    return serie

for df in [train, test]:
    df = transform_features(df)

# apply log transformation for the price
# train['price'] = target_transform(train['price'])

# set training data
X_train = train.copy()
y_train = X_train.pop('price')
X_test = test.copy()

if not SUBMIT:
    y_test = X_test.pop('price')
else:
    y_test = None

# %% Model

regressors = {
    'LGBMRegressor1': LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='gbdt'),
    # 'LGBMRegressor2': LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='dart'),
    # 'LGBMRegressor3': LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='goss'),
    # 'LGBMRegressor4': LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='rf', subsample=.632, subsample_freq=1),
    # 'LGBMRegressor5': LGBMRegressor(random_state=SEED, n_jobs=-1, class_weight='balanced'),
    # 'LGBMRegressor6': LGBMRegressor(random_state=SEED, n_jobs=-1, subsample=0.7),
    # 'LGBMRegressor7': LGBMRegressor(random_state=SEED, n_jobs=-1, colsample_bytree=0.7),
    # 'LGBMRegressor8': LGBMRegressor(random_state=SEED, n_jobs=-1, subsample=0.7, colsample_bytree=0.7),
    # 'LGBMRegressor9': LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='dart', colsample_bytree=0.7),
    'XGBRegressor1': XGBRegressor(random_state=SEED, n_jobs=-1),
    # 'XGBRegressor2': XGBRegressor(random_state=SEED, n_jobs=-1, booster='dart'),
    # 'XGBRegressor3': XGBRegressor(random_state=SEED, n_jobs=-1, booster='gblinear'),
    # 'XGBRegressor4': XGBRegressor(random_state=SEED, n_jobs=-1, colsample_bytree=0.7),
    # 'XGBRegressor5': XGBRegressor(random_state=SEED, n_jobs=-1, subsample=0.7),
    'XGBRandomForestRegressor': XGBRFRegressor(random_state=SEED, n_jobs=-1),
    # 'CatBoostRegressor': CatBoostRegressor(random_state=SEED, silent=True),
    'RandomForestRegressor': RandomForestRegressor(random_state=SEED, n_jobs=-1),
    # 'ExtraTreesRegressor': ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
    # 'AdaBoostRegressor': AdaBoostRegressor(random_state=SEED),
    # 'GradientBoostingRegressor': GradientBoostingRegressor(random_state=SEED),
    # 'BaggingRegressor': BaggingRegressor(random_state=SEED, n_jobs=-1),
    # 'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1),
    # 'DecisionTreeRegressor': DecisionTreeRegressor(random_state=SEED),
    # 'GaussianProcessRegressor': GaussianProcessRegressor(random_state=SEED),
    # 'MLPRegressor1': MLPRegressor(random_state=SEED, max_iter=1000, activation='relu', solver='adam'),
    # 'MLPRegressor2': MLPRegressor(random_state=SEED, max_iter=1000, activation='relu', solver='lbfgs'),
    # 'MLPRegressor3': MLPRegressor(random_state=SEED, max_iter=5000, activation='tanh', solver='adam'),
    # 'MLPRegressor4': MLPRegressor(random_state=SEED, max_iter=1000, activation='tanh', solver='lbfgs'),
    # 'MLPRegressor5': MLPRegressor(random_state=SEED, max_iter=5000, activation='logistic', solver='adam'),
    # 'MLPRegressor6': MLPRegressor(random_state=SEED, max_iter=1000, activation='logistic', solver='lbfgs'),
    # 'MLPRegressor7': MLPRegressor(random_state=SEED, max_iter=5000, activation='identity', solver='adam'),
    # 'MLPRegressor8': MLPRegressor(random_state=SEED, max_iter=5000, activation='identity', solver='lbfgs'),
    'Ridge': Ridge(random_state=SEED),
    # 'SGDRegressor': SGDRegressor(random_state=SEED, max_iter=1000, tol=1e-3),
    # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=SEED, max_iter=1000, tol=1e-3),
    # 'Perceptron': Perceptron(random_state=SEED, max_iter=1000, tol=1e-3),
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(random_state=SEED),
    'ElasticNet': ElasticNet(random_state=SEED),
    # 'HuberRegressor': HuberRegressor(max_iter=1000),
    # 'BayesianRidge': BayesianRidge(),
    # 'ARDRegression': ARDRegression(),
    # 'TheilSenRegressor': TheilSenRegressor(random_state=SEED),
    # 'RANSACRegressor': RANSACRegressor(random_state=SEED),
    # 'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(normalize=False),
    # 'Lars': Lars(),
    # 'LassoLars': LassoLars(),
    # 'LassoLarsIC': LassoLarsIC(normalize=False),
    # 'StackingRegressor': StackingRegressor(
    #         estimators=[
    #             ('LGBMRandomForestRegressor', LGBMRegressor(random_state=SEED, n_jobs=-1, boosting_type='rf', subsample=.632, subsample_freq=1)),
    #             ('XGBRandomForestRegressor', XGBRFRegressor(random_state=SEED, n_jobs=-1)),
    #             ('RandomForestRegressor', RandomForestRegressor(random_state=SEED, n_jobs=-1)),
    #             # ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=SEED, n_jobs=-1))
    #             ], 
    #         final_estimator=Ridge(random_state=SEED),
    #         # cv=cv,
    #         # n_jobs=-1,
    #         )
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

for model_name, model in regressors.items():
    t0 = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    print(f'{model_name}: {scores.mean():.4f} ± {scores.std():.4f}, Time: {time.time() - t0:.2f} seconds')

# %% Submission
y_pred_test = test_predictions.mean(axis=1).round().astype(int)

sub = pd.read_csv('submissions/sample_submission.csv')
sub['quality'] = y_pred_test
now = time.strftime("%Y-%m-%d %H_%M_%S")
sub.to_csv(f'submissions/submission{now}.csv', index=False)
# Copy the leaked values from the original dataset before submitting
# Transform the price column back to the original scale
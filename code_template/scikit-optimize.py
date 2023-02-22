# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")
# Classifiers
import lightgbm as lgb
# Model selection
from sklearn.model_selection import KFold
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

# Loading data 
X = pd.read_csv("../input/30-days-of-ml/train.csv")
X_test = pd.read_csv("../input/30-days-of-ml/test.csv")
# Preparing data as a tabular matrix
y = X.target
X = X.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# Dealing with categorical data
categoricals = [item for item in X.columns if 'cat' in item]
cat_values = np.unique(X[categoricals].values)
cat_dict = dict(zip(cat_values, range(len(cat_values))))
X[categoricals] = X[categoricals].replace(cat_dict).astype('category')
X_test[categoricals] = X_test[categoricals].replace(cat_dict).astype('category')

# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performance of optimizers
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds, candidates checked: %d, best CV            score: %.3f" + u" \u00B1"+" %.3f") % 
                             (time() - start,
                             len(optimizer.cv_results_['params']),
                             best_score, 
                             best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

# Setting the scoring function
scoring = make_scorer(partial(mean_squared_error, squared=False),
                      greater_is_better=False)
# Setting the validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# Setting the basic regressor
reg = lgb.LGBMRegressor(boosting_type='gbdt',
                        metric='rmse',
                        objective='regression',
                        n_jobs=1, 
                        verbose=-1,
                        random_state=0)

# Setting the search space
search_spaces = {
     
     # Boosting learning rate
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
     
     # Number of boosted trees to fit
    'n_estimators': Integer(30, 5000),
     
     # Maximum tree leaves for base learners
    'num_leaves': Integer(2, 512),
    
     # Maximum tree depth for base learners
    'max_depth': Integer(-1, 256),
     # Minimal number of data in one leaf
    'min_child_samples': Integer(1, 256),
     # Max number of bins buckets
    'max_bin': Integer(100, 1000),
     # Subsample ratio of the training instance 
    'subsample': Real(0.01, 1.0, 'uniform'),
     # Frequency of subsample 
    'subsample_freq': Integer(0, 10),
                
     # Subsample ratio of columns
    'colsample_bytree': Real(0.01, 1.0, 'uniform'), 
    
     # Minimum sum of instance weight
    'min_child_weight': Real(0.01, 10.0, 'uniform'),
   
     # L2 regularization
    'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),
         
     # L1 regularization
    'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),
   }

# Wrapping everything up into the Bayesian optimizer
opt = BayesSearchCV(estimator=reg,
                    search_spaces=search_spaces,
                    scoring=scoring,
                    cv=kf,
                    n_iter=60,           # max number of trials
                    n_jobs=-1,           # number of jobs
                    iid=False,         
                    # if not iid it optimizes on the cv score
                    return_train_score=False,
                    refit=False,  
                    # Gaussian Processes (GP) 
                    optimizer_kwargs={'base_estimator': 'GP'},
                    # random state for replicability
                    random_state=0)

# Running the optimizer
overdone_control = DeltaYStopper(delta=0.0001)
# We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60 * 60 * 6)
# We impose a time limit (6 hours)
best_params = report_perf(opt, X, y,'LightGBM_regression', 
                          callbacks=[overdone_control, time_limit_control])


# SECOND EXAMPLE with custom Bayesian optimization search
# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")
# Classifier/Regressor
from xgboost import XGBRegressor
# Model selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, forest_minimize
from skopt import gbrt_minimize, dummy_minimize
# Decorator to convert a list of parameters to named arguments
from skopt.utils import use_named_args 
# Data processing
from sklearn.preprocessing import OrdinalEncoder

# Loading data 
X_train = pd.read_csv("../input/30-days-of-ml/train.csv")
X_test = pd.read_csv("../input/30-days-of-ml/test.csv")
# Preparing data as a tabular matrix
y_train = X_train.target
X_train = X_train.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# Pointing out categorical features
categoricals = [item for item in X_train.columns if 'cat' in item]
# Dealing with categorical data using OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X_train[categoricals] = ordinal_encoder.fit_transform(X_train[categoricals])
X_test[categoricals] = ordinal_encoder.transform(X_test[categoricals])

# Setting the scoring function
scoring = partial(mean_squared_error, squared=False)
# Setting the cv strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# Setting the search space
space = [Real(0.01, 1.0, 'uniform', name='learning_rate'),
         Integer(1, 8, name='max_depth'),
         Real(0.1, 1.0, 'uniform', name='subsample'),
         # Subsample ratio of columns by tree
         Real(0.1, 1.0, 'uniform', name='colsample_bytree'),  
         # L2 regularization
         Real(0, 100., 'uniform', name='reg_lambda'),
         # L1 regularization
         Real(0, 100., 'uniform', name='reg_alpha'),
         # minimum sum of instance weight (hessian)  
         Real(1, 30, 'uniform', name='min_child_weight')
         ]
model = XGBRegressor(n_estimators=10_000, 
                     booster='gbtree', random_state=0)

# The objective function to be minimized
def make_objective(model, X, y, space, cv, scoring, validation=0.2):
    # This decorator converts your objective function 
    # with named arguments into one that accepts a list as argument,
    # while doing the conversion automatically.
    @use_named_args(space) 
    def objective(**params):
        model.set_params(**params)
        print("\nTesting: ", params)
        validation_scores = list()
        for k, (train_index, test_index) in enumerate(kf.split(X, y)):
            val_index = list()
            train_examples = int(train_examples * (1 - validation))
            train_index, val_index = (train_index[:train_examples], 
                                      train_index[train_examples:])
            
            start_time = time()
            model.fit(X.iloc[train_index,:], y[train_index],
                      early_stopping_rounds=50,
                      eval_set=[(X.iloc[val_index,:], y[val_index])], 
                      verbose=0
                    )
            end_time = time()
            
            rounds = model.best_iteration
            
            test_preds = model.predict(X.iloc[test_index,:])
            test_score = scoring(y[test_index], test_preds)
            print(f"CV Fold {k+1} rmse:{test_score:0.5f}-{rounds} 
                  rounds - it took {end_time-start_time:0.0f} secs")
            validation_scores.append(test_score)
            if len(history[k]) >= 10:
                threshold = np.percentile(history[k], q=25)
                if test_score > threshold:
                    print(f"Early stopping for under-performing fold: 
                          threshold is {threshold:0.5f}")
                    return np.mean(validation_scores)
                
            history[k].append(test_score)
        return np.mean(validation_scores)
    return objective

objective = make_objective(model,
                           X_train, y_train,
                           space=space,
                           cv=kf,
                           scoring=scoring)

def onstep(res):
    global counter
    x0 = res.x_iters   # List of input points
    y0 = res.func_vals # Evaluation of input points
    print('Last eval: ', x0[-1], 
          ' - Score ', y0[-1])
    print('Current iter: ', counter, 
          ' - Best Score ', res.fun, 
          ' - Best Args: ', res.x)
    # Saving a checkpoint to disk
    joblib.dump((x0, y0), 'checkpoint.pkl') 
    counter += 1
    
counter = 0
history = {i:list() for i in range(5)}
used_time = 0
gp_round = dummy_minimize(func=objective,
                          dimensions=space,
                          n_calls=30,
                          callback=[onstep],
                          random_state=0)

x0, y0 = joblib.load('checkpoint.pkl')
print(len(x0))

x0, y0 = joblib.load('checkpoint.pkl')
gp_round = gp_minimize(func=objective,
                       x0=x0,    # already examined values for x
                       y0=y0,    # observed values for x0
                       dimensions=space,
                       acq_func='gp_hedge',
                       n_calls=30,
                       n_initial_points=0,
                       callback=[onstep],
                       random_state=0)

x0, y0 = joblib.load('checkpoint.pkl')
print(f"Best score: {gp_round.fun:0.5f}")
print("Best hyperparameters:")
for sp, x in zip(gp_round.space, gp_round.x):
    print(f"{sp.name:25} : {x}")
    
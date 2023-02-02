# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, StratifiedKFold
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
train = pd.read_csv('train.csv')
train.drop('id', axis=1, inplace=True)
orig_train = pd.read_csv('creditcard.csv')
X_test = pd.read_csv('test.csv')
X_test.drop('id', axis=1, inplace=True)

train = pd.concat([train, orig_train]).reset_index(drop=True)

# Create a rolling difference feature on the time
def add_time_diff(df):
    df_time = df['Time'].copy().drop_duplicates().to_frame()
    df_time['time_diff'] = df_time['Time'].diff().fillna(0)
    df = pd.merge(df, df_time, how='left', on='Time')
    return df['time_diff']

seconds_per_day = 3600*24
for df in [train, X_test]:
    # df['time_diff'] = add_time_diff(df)
    df['hour'] = df['Time'] % (24 * 3600) // 3600
    df['day'] = (df['Time'] // (24 * 3600)) % 7
    df = df.drop(['Time'], axis=1)
    df['a0'] = df.Amount == 0
    df['a1'] = df.Amount == 1
    # df['V20_div_Amount'] = df.V20 / df.Amount
    # df['V23_div_Amount'] = df.V23 / df.Amount
    # df['V27_div_28'] = df.V27 / df.V28
    # df['V20_div_Amount_div_V27_div_28'] = df['V20_div_Amount'] / df['V27_div_28']
    # df['V23_div_Amount_div_V27_div_28'] = df['V23_div_Amount'] / df['V27_div_28']
    
# train, val = train_test_split(train, test_size=0.2, random_state=42)

X_train = train.drop(['Class'], axis=1)
y_train = train['Class']

# X_val = val.drop(['Class'], axis=1)
# y_val = val['Class']

# %%
# all sklearn models have a predict_proba method
models = {
    # "KNN": KNeighborsClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0),
    "CatBoost": CatBoostClassifier(verbose=False, loss_function='Logloss', eval_metric='AUC'),
    "LightGBM": LGBMClassifier(objective='binary'),
    "Extra Trees": ExtraTreesClassifier(n_jobs=-1),
    # "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    # "Decision Tree": DecisionTreeClassifier(),
    # "Gaussian Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Support Vector Machine": SVC(probability=True),
    # "Linear Support Vector Machine": LinearSVC(),
    # "Neural Network1": MLPClassifier(max_iter=5000, activation='relu', solver='adam', random_state=42),
    # "Neural Network2": MLPClassifier(max_iter=5000, activation='relu', solver='sgd', random_state=42),
    # "Neural Network3": MLPClassifier(max_iter=5000, activation='tanh', solver='adam', random_state=42),
    "Neural Network4": MLPClassifier(max_iter=5000, activation='tanh', solver='sgd'),
    # "Neural Network5": MLPClassifier(max_iter=5000, activation='logistic', solver='adam', random_state=42),
    "Neural Network6": MLPClassifier(max_iter=5000, activation='logistic', solver='sgd'),
    # "Neural Network7": MLPClassifier(max_iter=5000, activation='identity', solver='adam', random_state=42),
    # "Neural Network8": MLPClassifier(max_iter=5000, activation='identity', solver='sgd', random_state=42),
    # "Ridge Classifier": RidgeClassifier(),
    # "Ridge Classifier with CV": RidgeClassifierCV(),
    # "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
    # "SGD Classifier": SGDClassifier(loss='modified_huber'),
    # "Bernoulli Naive Bayes": BernoulliNB(),
    "Calibrated Classifier LSVC": CalibratedClassifierCV(LinearSVC(), n_jobs=-1),
    "Calibrated Classifier Ridge": CalibratedClassifierCV(RidgeClassifier(), n_jobs=-1),
    # "Bagging Classifier": BaggingClassifier(),
    # "Multinomial Naive Bayes": MultinomialNB(),    
}

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                    columns=X_train.columns, 
                                    index=X_train.index)
# X_val_mm = pd.DataFrame(mm_scaler.transform(X_val),
#                             columns=X_val.columns,
#                             index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                            columns=X_test.columns,
                            index=X_test.index)

# %%
sample_size = 2000
predictions_assembled_train = []
predictions_assembled_val = []
predictions_assembled_test = []
df_pred_test_index = pd.DataFrame()
for name, model in models.items():
    t0 = time.time()
    pred_test_index = pd.Series()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        
        # Get a subset of the training set
        X = X_train_scaled.iloc[train_index]
        X_test_index = X_train_scaled.iloc[test_index]
        # X_val_ = X_val_mm
        X_test_ = X_test_scaled
        y = y_train.iloc[train_index]
        
        # Create a balanced training set
        X_balanced = pd.concat([X[y == 0].sample(sample_size, replace=True),
                                X[y == 1].sample(sample_size, replace=True)])

        y_balanced = y.loc[X_balanced.index]
        
        # Fit the model
        model.fit(X_balanced, y_balanced)
        
        # Evaluate the model's performance on train and validation sets
        train_roc_score = roc_auc_score(y_train.iloc[test_index],model.predict_proba(X_test_index)[:, 1])
        print(f"Model {name}, Fold {i}, test_index ROC score: {train_roc_score}")
        
        # Save the predictions
        # predictions_assembled_val.append(model.predict_proba(X_val_)[:, 1])
        predictions_assembled_test.append(model.predict_proba(X_test_)[:, 1])
        pred_test_index = pd.concat([pred_test_index, pd.Series(model.predict_proba(X_test_index)[:, 1], index=test_index)])
    
    t1 = time.time()
    print(f"{name} fit time: {t1 - t0:.2f}s")
    df_pred_test_index[name] = pred_test_index

# print("Mean AUC on validation set:")
# print(roc_auc_score(y_val, pd.DataFrame(predictions_assembled_val).T.mean(axis=1)))
y_pred_test = pd.DataFrame(predictions_assembled_test).T.mean(axis=1)

# %%
df_pred_train = pd.DataFrame()
df_pred = pd.DataFrame()
for name, model in models.items():
    print(name)
    df_pred_train[name] = model.predict_proba(X_train_scaled)[:, 1]
    df_pred[name] = model.predict_proba(X_test_scaled)[:, 1]
    
# %%
final_estimator = LogisticRegression(class_weight='balanced', penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)
final_estimator.fit(df_pred_train, y_train)
y_pred_test = final_estimator.predict(df_pred)

# %%
sub = pd.read_csv('sample_submission.csv')
sub['Class'] = y_pred_test
now = time.strftime("%Y-%m-%d %H_%M_%S")
sub.to_csv(f'submission{now}.csv', index=False)
# %%

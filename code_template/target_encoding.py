# from the Kaggle book
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncode(BaseEstimator, TransformerMixin):
    
    def __init__(self, categories='auto', k=1, f=1, 
                 noise_level=0, random_state=None):
        if type(categories)==str and categories!='auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state
        
    def add_noise(self, series, noise_level):
        return series * (1 + noise_level *   
                         np.random.randn(len(series)))
        
    def fit(self, X, y=None):
        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target']
                       .agg(['mean', 'count']))
            # Compute smoothing 
            smoothing = (1 / (1 + np.exp(-(avg['count'] - self.k) /                 
                         self.f)))
            # The bigger the count the less full_avg is accounted
            self.encodings[variable] = dict(self.prior * (1 -  
                             smoothing) + avg['mean'] * smoothing)
            
        return self
    
    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable], 
                                 inplace=True)
            unknown_value = {value:self.prior for value in 
                             X[variable].unique() 
                             if value not in 
                             self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                Xt[variable].replace(unknown_value, inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable], 
                                              self.noise_level)
        return Xt
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
"""  
The input parameters of the function are:

categories: The column names of the features you want to target-encode. You can leave 'auto' on and 
the class will pick the object strings.
k (int): Minimum number of samples to take a category average into account.
f (int): Smoothing effect to balance the category average versus the prior probability, or the mean 
value relative to all the training examples.
noise_level: The amount of noise you want to add to the target encoding in order to avoid 
overfitting. Start with very small numbers.
random_state: The reproducibility seed in order to replicate the same target encoding when 
noise_level > 0.

Notice the presence of the k and the f parameters. In fact, for a level i of a categorical feature, 
we are looking for an approximate value that can help us better predict the target using a single 
encoded variable. Replacing the level with the observed conditional probability could be the 
solution, but doesn’t work well for levels with few observations. The solution is to blend the 
observed posterior probability on that level (the probability of the target given a certain value 
of the encoded feature) with the a priori probability (the probability of the target observed on 
the entire sample) using a lambda factor. This is called the empirical Bayesian approach.

In practical terms, we are using a function to determine if, for a given level of a categorical 
variable, we are going to use the conditional target value, the average target value, or a blend of 
the two. This is dictated by the lambda factor, which, for a fixed k parameter (usually it has a 
unit value, implying a minimum cell frequency of two samples) has different output values depending 
on the f value that we choose.
"""
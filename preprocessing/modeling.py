import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
import time

# Full dataset
df = pd.read_csv('data/cleandata/FoodAccessCleaned.csv')

# Potential supervisors
# There some population counts that are 0 < pop < 1
# Rounding up so we don't have negative values when we log transform the supervisors
supervisors = ['LAPOP1_20','LAPOP1_10','LAPOP05_10','LALOWI1_10','LALOWI05_10','LALOWI1_20', 'lapophalf']
df[supervisors] = df[supervisors].apply(np.rint,axis=1)

# Dataset without potential supervisors
df_learn = df[df.columns[~df.columns.isin(supervisors)]]

# Creating lists of the cat and quant vars for later use in
# in the ColumnTransformer
quant_vars = df_learn.select_dtypes(include=['int64','float64']).columns.to_list()
factor_vars = df.select_dtypes(include=['datetime64','timedelta','object','category']).columns.to_list()

unique_counts = df[quant_vars].nunique().to_dict()
sus_num = [name for name,count in unique_counts.items() if count >= 10]
sus_cat = [name for name,count in unique_counts.items() if count < 10]

# Final list of cat vars (factor_vars) and quantitative vars (quant_vars)
factor_vars += sus_cat
quant_vars = sus_num

# TODO split into training and test set and then create pipeline object for all transformations
X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, df[supervisors[0]],
                     test_size=0.20,
                     stratify=df_learn['State'])

X_train, X_val, y_train, y_val =\
    train_test_split(X_train, y_train,
                     test_size=0.4,
                     stratify=X_train['State'])


transformer = ColumnTransformer([('minmax', MinMaxScaler() , quant_vars ),
                                 ('cat'   , OneHotEncoder(sparse=False), ['State']    ) ])

# Grid for l1_ratio (i.e. the ratio of lasso:ridge)
# I'm letting the default ElasticnetCV search figure out the amount of penalty
# so I can't include 0 in the grid (otherwise estimation breaks down) so
# I just chose a really small decimal
l1_grid  = [0.001,0.1,0.25,0.5,0.7,0.9,1]

# n_jobs = -1 means use all cores in computer
elastic_net = ElasticNetCV(l1_ratio=l1_grid, cv = 10,
                           n_jobs=None, max_iter=100000)

pipe = Pipeline( [('transform',  transformer),
                  ('elasticnet', elastic_net ) ])

pipe.fit(X_val,np.log(y_val,where=(y_val != 0.0) ))
val_coefs = pipe['elasticnet'].coef_

pipe.set_params(elasticnet__alphas=[pipe[1].alpha_])
pipe.set_params(elasticnet__l1_ratio=pipe[1].l1_ratio_)

pipe.fit(X_train,y_train)
train_coefs = pipe['elasticnet'].coef_

pipe.predict(X_test)
print(pipe.score(X_test,y_test))


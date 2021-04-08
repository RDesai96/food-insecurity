import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error as mse

# Full dataset
df = pd.read_csv('data/cleandata/FoodAccessCleaned.csv')

# Potential supervisors
# There some population counts that are 0 < pop < 1
# Rounding up so we don't have negative values when we log transform the supervisors
supervisors = ['LAPOP1_20','LAPOP1_10','LAPOP05_10','LALOWI1_10','LALOWI05_10','LALOWI1_20', 'lapophalf']
# df[supervisors] = df[supervisors].apply(np.rint,axis=1)

# Dataset without potential supervisors
df_learn = df[df.columns[~df.columns.isin(supervisors)]]

# Creating lists of the cat and quant vars for later use in
# in the ColumnTransformer
quant_vars = df_learn.select_dtypes(include=['int64','float64']).columns.to_list()
factor_vars = df.select_dtypes(include=['datetime64','timedelta','object','category']).columns.to_list()

unique_counts = df[quant_vars].nunique().to_dict()
sus_num = [name for name,count in unique_counts.items() if count > 10]
sus_cat = [name for name,count in unique_counts.items() if count <= 10]

# Final list of cat vars (factor_vars) and quantitative vars (quant_vars)
factor_vars += sus_cat
quant_vars = sus_num

# Split into train, val, and test sets using stratified by census tract population bins
supervisor = df[supervisors[1]]
# supervisor = np.log(supervisor, where=(supervisor != 0.0))

X_train, X_test, y_train, y_test = \
    train_test_split(df_learn, supervisor,
                     test_size=0.20,
                     stratify=df_learn['POP2010_bins'])

X_train, X_val, y_train, y_val =\
    train_test_split(X_train, y_train, test_size=0.4,stratify=X_train['POP2010_bins'])

# Dropping tract populations bins before proceeding with model fitting
X_train = X_train.drop('POP2010_bins',axis=1).copy()
X_val = X_val.drop('POP2010_bins',axis=1).copy()
X_test = X_test.drop('POP2010_bins',axis=1).copy()

transformer = ColumnTransformer([('minmax', StandardScaler() , quant_vars ),
                                 ('cat'   , OneHotEncoder(), ['State']    ) ])


# Grid for l1_ratio (i.e. the ratio of lasso:ridge)
# I'm letting the default ElasticnetCV search figure out the amount of penalty
# so I can't include 0 in the grid (otherwise estimation breaks down) so
# I just chose a really small decimal in place of 0
l1_grid  = [0.001,0.1,0.25,0.5,0.7,0.9,1]

# n_jobs = -1 means use all cores in computer
elastic_net = ElasticNetCV(l1_ratio=l1_grid, cv = 10,
                           n_jobs=None, max_iter=100000, random_state=57)

pipe = Pipeline( [('transform',  transformer),
                  ('elasticnet', elastic_net ) ])

# Converting all feature matrices to numpy arrays, as
# recommended in scikit-learn doc to avoid unnecessary copying

pipe.fit(X_val, y_val)
val_coefs = pipe['elasticnet'].coef_

print(pipe[1].alpha_)
print(pipe[1].l1_ratio_)

pipe.set_params(elasticnet__alphas=[pipe[1].alpha_])
pipe.set_params(elasticnet__l1_ratio=pipe[1].l1_ratio_)

pipe.fit(X_train,y_train)
train_coefs = pipe['elasticnet'].coef_

y_pred = pipe.predict(X_test)
print('Rsquared: ' + str(pipe.score(X_test,y_test)))

MSE = mse(y_test,y_pred)
RMSE = np.sqrt(MSE)
print('MSE: ' + str(MSE) )
print('RMSE: ' + str(RMSE))


# MSE = (math.e**mse(y_test,y_pred))
# RMSE = np.sqrt(MSE)
# print('MSE: ' + str(MSE))
# print('RMSE: ' + str(RMSE))

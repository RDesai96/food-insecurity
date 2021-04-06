import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
import time

# Potential supervisors
supervisors = ['LAPOP1_10','LAPOP05_10','LAPOP1_20','LALOWI1_10',
               'LALOWI05_10','LALOWI1_20', 'lapophalf']
# Full dataset
df = pd.read_csv('data/cleandata/FoodAccessCleaned.csv')

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
                                 ('cat'   , OneHotEncoder(), ['State']    ) ])

# Grid for l1_ratio (i.e. the ratio of lasso:ridge)
# I'm letting the default ElasticnetCV search figure out the amount of penalty
# so I can't include 0 in the grid (otherwise estimation breaks down) so
# I just chose a really small decimal
l1_grid  = [0.0001,0.2,0.4,0.6,0.8,1]

# n_jobs = -1 means use all cores in computer
elastic_net = ElasticNetCV(l1_ratio=l1_grid, cv = 10, n_jobs=None)

pipe = Pipeline( [('transform',  transformer),
                  ('elasticnet', elastic_net ) ])

t0 = time.time()
pipe.fit(X_val,y_val)
t1 = time.time()
total_time = t1-t0


coefs = pipe[1].coef_
pipe['transform'].transformers_[1][1].categories_

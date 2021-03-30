import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelBinarizer

# NOTE: you will have to insert your own data-paths here
rawdata = '/Users/rajandesai/PycharmProjects/data/rawdata/FoodAccessRaw.csv'
datalabels = '/Users/rajandesai/PycharmProjects/data/rawdata/FoodAccessLabels.csv'

# Read in df
df = pd.read_csv(rawdata)
df_labels = df.columns.to_list()

# Checking for missing values
# print(df.isna().sum().sum())
# print(df.isnull().sum().sum())

# Create a list of numeric vars and a list of all other dtypes
quant_vars = df.select_dtypes(include=['int64','float64']).columns.to_list()
factor_vars = df.select_dtypes(include=['datetime64','timedelta','object','category']).columns.to_list()

# Identifying possible categorical variables encoded as numeric by looking at columns with <10 unique values
unique_counts = df[quant_vars].nunique().to_dict()
sus_cat = [name for name,count in unique_counts.items() if count < 10]
sus_num = [name for name,count in unique_counts.items() if count >= 10]

# Removing CensusTract (it's just numeric key). Combing lists of cat variables
sus_num.remove('CensusTract')
sus_cat += factor_vars
sus_cat.remove('County')

# Removing potential supervisor variables from sus_cat
supervisors = df_labels[25:32]
sus_cat = [val for val in sus_cat if val not in supervisors]
sus_num = [val for val in sus_num if val not in supervisors]

# Creating a quant vars df and a cat vars df
df_quant = df[sus_num].copy()
df_factor = df[sus_cat].copy()

# Create dummy vars for "State"
lb = LabelBinarizer(neg_label=0,pos_label=1)
dummy_vars = lb.fit_transform(df_factor['State'])
dummy_df = pd.DataFrame(dummy_vars, columns = lb.classes_)

# Append dummy vars to df_factor
df_factor.drop('State', axis = 1, inplace=True)
df_factor = pd.concat([df_factor,dummy_df],axis=1)

#


# Scaling quant
# #Replace all NAs w/ 0s in sus_num columns
# dict_na = {val:0 for val in df_cols}
# df_quant = df_quant.fillna(dict_na)
# print(df_quant.isna().sum())

# Exam


sn.heatmap(df_quant[sus_num].corr(), cmap = 'Greens')
plt.title('U.S. Food Insecurity Indicators (from 2010 Census)')
plt.show()
# df.hist(bins=50, figsize=(20,15))

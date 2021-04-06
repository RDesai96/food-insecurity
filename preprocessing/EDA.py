import pandas as pd

# Below is the code I used for EDA (Exploratory Data Analysis)
# This dataset did not really require any cleaning, you can use the
# code below to get a better sense of the data.

# Download data at: https://www.kaggle.com/tcrammond/food-access-and-food-deserts
rawdata = 'data/rawdata/FoodAccessRaw.csv'
datalabels = 'data/rawdata/FoodAccessLabels.csv'

df = pd.read_csv(rawdata)
df_labels = df.columns.to_list()

# Removing CensusTract (it's just a numeric key) and County
df.drop(['CensusTract','County'],axis=1,inplace=True)

# Checking for missing values, there are no missing values
print(df.isna().sum().sum())
print(df.isnull().sum().sum())

# Create a list of numeric vars and a list of all other dtypes
quant_vars = df.select_dtypes(include=['int64','float64']).columns.to_list()
factor_vars = df.select_dtypes(include=['datetime64','timedelta','object','category']).columns.to_list()

# Examining possible categorical variables encoded as numeric by looking at columns with <10 unique values
unique_counts = df[quant_vars].nunique().to_dict()
sus_cat = [name for name,count in unique_counts.items() if count < 10]
sus_num = [name for name,count in unique_counts.items() if count >= 10]

# Exporting df for modeling
df.to_csv('data/cleandata/FoodAccessCleaned.csv',sep=',', index=False)


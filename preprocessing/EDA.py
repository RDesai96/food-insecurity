import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
supervisors = ['LAPOP1_20','LAPOP1_10','LAPOP05_10','LALOWI1_10','LALOWI05_10','LALOWI1_20', 'lapophalf']

# Plotting the potential supervisors.
# All highly skewed to the right since most census tracts have <100 people
def draw_histograms(df, variables, n_rows, n_cols, bin_size):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=bin_size,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

draw_histograms(df[supervisors], supervisors, 3, 3, 10)

# Log transforming gets rid of skewness
# Rounding up all population counts since fractional and
# population counts <0 don't make sense before transforming
df_log = df[supervisors].apply(np.rint,axis=1)
df_log = np.log(df_log.replace(0,np.nan))
df_log = df_log.fillna(0)
draw_histograms(df_log, supervisors, 3, 3, 10)


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


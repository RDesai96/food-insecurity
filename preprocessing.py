import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# Reading data and separating based on dtype
df = pd.read_csv('/Users/rajandesai/Desktop/food_access_research_atlas.csv')
df.head(600).to_html('Food_Des.html')
df_cols = df.columns.to_list()

#Overview of the dataset
print(df.info())

#Separating dataframe into quant and qual datatypes
df_quant  = df.select_dtypes(include=['int64','float64','datetime64','timedelta']).copy()
df_factor = df.select_dtypes(include=['bool','object','category']).copy()


#Checking for missing values
# print(df.isna().sum().sum())
# print(df.isnull().sum().sum())

#Identifying possible categorical variables encoded as numeric by looking at columns wiht <10 unique values
unique_counts = df_quant.nunique().to_dict()
sus_cat = [name for name,count in unique_counts.items() if count < 10]
sus_num = [name for name,count in unique_counts.items() if count >= 10]

#Refining df_quant and df_factor
sus_num.remove('CensusTract')


# #Replace all NAs w/ 0s in sus_num columns
# dict_na = {val:0 for val in df_cols}
# df_quant = df_quant.fillna(dict_na)
# print(df_quant.isna().sum())

#Exam


sn.heatmap(df_quant[sus_num].corr(), cmap = 'Greens')
plt.title('U.S. Food Insecurity Indicators (from 2010 Census)')
plt.show()
# df.hist(bins=50, figsize=(20,15))

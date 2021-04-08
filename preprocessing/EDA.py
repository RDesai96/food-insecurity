import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

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

# df of just the features
df_features = df[df.columns[~df.columns.isin(supervisors)]]

# Plotting histograms of the data
def draw_histograms(df, variables, n_rows, n_cols, bin_size=100):
    """Draws a grid of 100 bin histograms with n_rows x n_cols"""
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=bin_size,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# Creating a pdf file with all the histograms for all the features
def hist_pdf(df, filename='Feature_hists', bins=100):
    """Creates a pdf file of 9 x 9 100 bin histograms on each page out of your feature df
    and saves to filename"""
    pp = PdfPages(filename + '.pdf')
    ceil_hist = (df.shape[1] // 9)*9
    if ceil_hist <= 9:
        draw_histograms(df,df.columns,3,3,bin_size=bins)
        pp.savefig()
    else:
    for i in range(9,df.shape[1]+9,9):
        if i <= ceil_hist:
            draw_histograms(df.iloc[:,(i-9):i], df.columns[(i-9):i], 3, 3, bin_size=bins)
            pp.savefig()
        elif i > ceil_hist:
            i = df.shape[1]
            diff = df.shape[1] - ceil_hist
            draw_histograms(df.iloc[:,(i-diff):i], df.columns[(i-diff):i], 3, 3, bin_size=bins)
            pp.savefig()
    pp.close()
hist_pdf(df_features, filename='preprocessing/Plots/Feature_hists')

# histograms of the potential supervisors.
# All highly skewed to the right since there are lot of tracts with ~0 people
hist_pdf(df[supervisors],filename='preprocessing/Plots/Sup_hists')

# Log transforming gets rid of skewness
# Rounding up all population counts since fractional and
# population counts <0 don't make sense before transforming
df_log = df[supervisors].apply(np.rint,axis=1)
df_log = np.log(df_log.replace(0,np.nan))
df_log = df_log.fillna(0)

# histograms of the transformed supervisors
hist_pdf(df_log,filename='preprocessing/Plots/Sup_log_hists')

# Correlation matrix with all the transformed supervisors
def sborn_heatmap(df,filename='plot',title='Plot'):
    """Plots and saves a seaborn heatmap to filename as a png"""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.savefig(filename + '.png')
sborn_heatmap(df_log,filename='preprocessing/Plots/sup_log_corrs',
              title='Transformed Supervisors')

# Correlation matrix with all the features
sborn_heatmap(df_features,filename='preprocessing/Plots/feature_corrs',
              title='Use FoodAccessLabel.csv as reference to identify unlabeled features')

# Checking for missing values, there are no missing values
print(df.isna().sum().sum())
print(df.isnull().sum().sum())

# Create a list of numeric vars and a list of all other dtypes
def dtype_split(df):
    """Returns a list of quantitative variables (pandas.dtype = int64,float64) and a list of
     categorical variables (pandas.dtype = datetime64,timedelta,object,category)."""
    quant_vars = df.select_dtypes(include=['int64','float64']).columns.to_list()
    factor_vars = df.select_dtypes(include=['datetime64','timedelta','object','category']).columns.to_list()
    return quant_vars, factor_vars
quant_vars, factor_vars = dtype_split(df)

# Examining possible categorical variables encoded as numeric by looking at columns with <10 unique values
def quant_examine(df, limit=10):
    """Returns a list of suspected categorical vars and numeric vars
    depending on if df.unique_counts > limit (default limit = 10) """
    unique_counts = df[quant_vars].nunique().to_dict()
    sus_cat = [name for name,count in unique_counts.items() if count < limit]
    sus_num = [name for name,count in unique_counts.items() if count >= limit]
    return sus_cat, sus_num
sus_cat, sus_num = quant_examine(df[quant_vars])

# Encoding new column of bins based on tract population
# Will be useful later for stratified data splitting
bin_labels = ['a','b','c','d','e','f','g','h','i','k']
df['POP2010_bins'] = pd.cut(df['POP2010'],
                            bins=[-np.inf,1000,2000,3000,4000,5000,6000,7000,8000,9000,np.inf],
                            labels=bin_labels,retbins=False)
df.to_csv('data/cleandata/FoodAccessCleaned.csv',sep=',', index=False)

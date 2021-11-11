
################### Import Libraries ###########################
import timeit
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from datetime import (date, timedelta, datetime)
from sklearn.metrics import (mean_absolute_error, mean_squared_error)

# for plotting
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

# Import Time Series libraries
from scipy.stats.distributions import chi2
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# misc
from datetime import datetime
from warnings import filterwarnings
from datetime import timedelta, date
from prophet import Prophet
from prophet.diagnostics import cross_validation
import prophet
from pycaret.regression import load_model, predict_model
sns.set_style('white')
filterwarnings('ignore')

###################### End of Import LIbraries ############################

###################### Start of Forecasting Frequency #####################

forecast_days=10
train_size_param=0.75
freq=365
###################### End of Forecasting Frequency #######################

###################### Start of Functions used for forecasting ############
###########################################################################


### Function for Outlier Detection ##
def flag_outlier(data,x):
    lower_limit  = np.mean(data['demand']) - np.std(data['demand']) *2 
    if x<lower_limit:
        return(lower_limit)
    upper_limit = np.mean(data['demand']) + np.std(data['demand']) *2
    if x>upper_limit:
        return(upper_limit)
    return(x)
###########################################################################
### Function for Outlier Detection ##
def flag_outlier1(data,x):
    lower_limit  = np.mean(data['demand']) - np.std(data['demand']) *3 
    if x<lower_limit:
        return(True)
    upper_limit = np.mean(data['demand']) + np.std(data['demand']) *3 
    if x>upper_limit:
        return(True)
    return(False)

#################### Start of Main Function ##############################
def load_data():
    print('Loading Data')
    #res1 = s3.get_object(Bucket=bucket_name, Key=f"df_iqr_iterations/20/{running_date}/forecast_datasets/related.csv")
    related_df = pd.read_csv("D:\\TTN\\Hyke\\Forecasting Project\\Datasets\\28-10-2021\\related.csv")
    print('Loaded related file')
    #res2 = s3.get_object(Bucket=bucket_name, Key=f"df_iqr_iterations/20/{running_date}/forecast_datasets/target.csv")
    target_df = pd.read_csv("D:\\TTN\\Hyke\\Forecasting Project\\Datasets\\28-10-2021\\target.csv")
    print('Loaded target file')
    #res3 = s3.get_object(Bucket=bucket_name,
    #                     Key=f"df_iqr_iterations/20/{running_date}/product_pareto_n_safety_stock/pareto_output.csv")
    pareto_df = pd.read_csv("D:\\TTN\\Hyke\\Forecasting Project\\Datasets\\28-10-2021\\pareto_output.csv",sep='~')
    print('Loaded Pareto file')

    # res4 = s3.get_object(Bucket=bucket_name,
    #                      Key=f"df_iqr_iterations/20/{running_date}/product_attributes/output.csv")
    # attributes = pd.read_csv(res4.get("Body"))
    print('Loaded attributes file')
    return target_df, pareto_df, related_df

df, pareto_df, related_df = load_data()

_country = 'UAE'
B_C_item_ids = pareto_df.loc[pareto_df['pareto_category'].isin(['B', 'C'])]
unq_B_C_item_ids = B_C_item_ids["item_id"].unique()
uae_items = [x for x in unq_B_C_item_ids if f'{_country}' in x[-9:]]

df = df.loc[df["item_id"].isin(uae_items)]
related_df = related_df.loc[related_df["item_id"].isin(uae_items)]
print(f'Total number of {_country} SKUs for forecasting : {len(uae_items)}')
# forecasted_date = df['timestamp'].max()
df = pd.merge(df, related_df, on=['item_id', 'timestamp'], how='inner')

### VIEW FEW INITIAL ROWS ###
df.head()

### CONVERT TIME STAMP TO DATETIME ###
df['date'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df.head()

### REPLACE 0 WITH NAN ###
df['demand']=df['demand'].replace(0,np.nan)

### COUNT NO. OF NAN VALUES PER ITEM ID ###
df2 = df.demand.isnull().groupby(df['item_id']).sum().astype(int).reset_index(name='count')

### COUNT TOTAL ROWS OF EACH ITEM ID ###
df3 = df['item_id'].value_counts()
df3=df3.to_frame()
df3.columns=['Total_Count']
df3['item_id'] = df3.index

##### Merge df2 with Null Count and df3 with Total Count #######
df2=pd.merge(df2,df3,on='item_id')
df2.columns=['item_id','0_count','Total_Count']

#### SORT VALUES ###
df2.sort_values(by=['Total_Count','0_count'],ascending=['False','True'],inplace=True)

### CALCULATE PERCENTAGE OF MISSING VALUES PER ITEM ID ###
df2['percentage']= (df2['0_count']/df2['Total_Count'])

#final_df=df.loc[df['item_id']=='AXMMOOPPA54DS64GB4GBSBLU_UAE08209',]
final_df =pd.merge(df,df2, on='item_id')
### SELECT PARTS HAVING MISSING VALUE PERCENTAGE BETWEEN 0 TO 40 ###
final_df=final_df.loc[(final_df['percentage']>0) & (final_df['percentage']<0.50),]
final_df = final_df.sort_values(['item_id', 'timestamp'], ascending=[True, True])

### GROUP ITEMS ###
item_groups = final_df.groupby('item_id')
print("Grouping Done !")
l_grouped = list(item_groups)
i=0

from pycaret.regression import *
from tqdm import tqdm
all_ts = final_df['item_id'].unique()

all_results = []
all_score_df = []
final_model = {}
count=0
i='AXMACHUA55033514_UAE08164'
for i in tqdm(all_ts):
    print("Running:",(count+1))
    count=count+1
    df_subset = final_df[final_df['item_id'] == i]
    last_index=df_subset.apply(pd.Series.last_valid_index)
    last_index=last_index['demand']
    ### DROP ROWS AFTER LAST INDEX DEMAND ###
    df_subset=df_subset.drop(df_subset.index[list(range(last_index,len(df_subset)))])
    
    #### Imputing MISSING VALUES BY INTERPOLATION METHOD#####
    check_df_impute_interpolation=df_subset.copy()
    #check_df_impute_interpolation.set_index('date')['demand'].interpolate(method="linear")
    check_df_impute_interpolation['demand']=check_df_impute_interpolation['demand'].interpolate(method="linear")
    check_df_impute_interpolation.fillna(0, inplace=True)
    check_df_impute_interpolation['demand']=check_df_impute_interpolation['demand'].apply(lambda x: flag_outlier(check_df_impute_interpolation,x))
    plt.plot(check_df_impute_interpolation['timestamp'],check_df_impute_interpolation['demand'])
    plt.show()

    ### Replacing 0 with 1 so that MAPE doesnt get infinite
    check_df_impute_interpolation['demand']=check_df_impute_interpolation['demand'].replace(0,1)
    
    #creating the train and validation set
    train = check_df_impute_interpolation[:int(0.75*(len(check_df_impute_interpolation)))]
    valid = check_df_impute_interpolation[int(0.75*(len(check_df_impute_interpolation))):]
    train= train.iloc[:,[0,1,2,3]]
    valid= valid.iloc[:,[0,1,2,3]]
    # initialize setup from pycaret.regression
    s = setup(train, target = 'demand', train_size = 0.75,
              data_split_shuffle = False, fold_strategy = 'timeseries', fold = 3,
              numeric_features = ['price'],
              silent = True, verbose = False, session_id = 123)
    
    # compare all models and select best one based on MAE
    best_model = compare_models(sort = 'MAE', verbose=True)
    
    # capture the compare result grid and store best model in list
    p = pull().iloc[0:1]
    p['time_series'] = str(i)
    all_results.append(p)
    
    # finalize model i.e. fit on entire data including test set
    f = finalize_model(best_model)
    
    
    #### Prediction #####
    p = predict_model(f, data=valid)
    all_score_df.append(p)
    
    # attach final model to a dictionary
    final_model[i] = f
    
    # save transformation pipeline and model as pickle file 
    #save_model(f, model_name='trained_models/' + str(i), verbose=False)
    
    
concat_results = pd.concat(all_results,axis=0)
concat_results.head()
#y_actual=valid['demand']
#valid=valid.iloc[:,[1,2,3]]

#all_score_df = []
#for i in tqdm(data['time_series'].unique()):
#    l = f#load_model('trained_models/' + str(i), verbose=False)
##    p = predict_model(l, data=valid)
##    p['time_series'] = i
#    all_score_df.append(p)
concat_df = pd.concat(all_score_df, axis=0)
concat_df.head()
concat_results.to_csv('D:\\TTN\\Hyke\\Forecasting Project\\Analysis\\28-10-2021\\Pycaret\\Best Technique.csv') 
   
   
concat_df.to_csv('D:\\TTN\\Hyke\\Forecasting Project\\Analysis\\28-10-2021\\Pycaret\\Results.csv') 
   
   





# create 12 month moving average
#cat1_sampled_D['MA12'] = cat1_sampled_D['demand'].rolling(12).mean()
#cat1_sampled_D['Date']=cat1_sampled_D.index


#cat1_sampled_D.plot(x="Date", y=["demand", "MA12"])
#plt.show()

#creating the train and validation set
##train = cat1_sampled_D[:int(0.75*(len(cat1_sampled_D)))]
#test = cat1_sampled_D[int(0.75*(len(cat1_sampled_D))):]

#train=train.iloc[:,[0,1]]
#test=test.iloc[:,[0,1]]

# import the regression module
#from pycaret.regression import *
# initialize setup
#s = setup(data = train, test_data = test, target = 'demand', fold_strategy = 'timeseries', numeric_features = ['price'], fold = 1, transform_target = True, session_id = 123)
#best = s.compare_models(sort = 'MAPE')
#prediction_holdout = predict_model(best);

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.datasets.co2 as co2



# First we read data

df_stores = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/stores_dataset.csv')
df_features = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/Features_dataset.csv', parse_dates = ['Date'])
df_sales = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/sales_dataset.csv', parse_dates = ['Date'])


# we delete unnecessary data
df_features = df_features[df_features.Date.dt.date <= df_sales.Date.dt.date.max()]

# data merging 1
df_all_1 = df_features.merge(df_sales, 'right', on = ['Date', 'Store', 'IsHoliday'])

# data merging 2
df_all = df_all_1.merge(df_stores, 'left', on = 'Store')
print('New Min date in features ds', df_all.Date.dt.date.min(), ',max: ', df_all.Date.dt.date.max())


#Finally, we change IsHoliday to a quantitative column, then set Date as the index.

df_all.IsHoliday = df_all.IsHoliday.map(lambda x: 1 if x == True else 0)


# check for missing values
print(df_all.isna().sum())

# group by date for modeling on all data
df_by_date = df_all.groupby('Date', as_index=False).agg({'Temperature': 'mean',
                                                        'Fuel_Price': 'mean',
                                                        'CPI': 'mean',
                                                        'Unemployment': 'mean', 
                                                        'Weekly_Sales': 'sum',
                                                        'IsHoliday': 'mean'})

#print(df_by_date.head())

# setting datetime format and making Date an index
df_by_date.Date = pd.to_datetime(df_by_date.Date, errors='coerce')
df_by_date.set_index('Date', inplace=True)

# resampling data weekly
df_by_date_new = df_by_date.resample('W').mean().fillna(method='bfill')
#

# checking for seasonality in time series
multi_plot = seasonal_decompose(df_by_date_new['Weekly_Sales'], model = 'add', extrapolate_trend='freq')

#plt.figure(figsize=(20,5))
#multi_plot.observed.plot(title = 'weekly sales')

#plt.figure(figsize=(20,5))
#multi_plot.seasonal.plot(title = 'seasonal')
#As it can be observed, the series is strongly influenced by the seasonal component

#Correlations

#plt.figure(figsize=(15,8))
#sns.heatmap(df_by_date_new.corr('spearman'), annot = True);

#there is no correlation worth mentioning


## drill down by store

## we want to group by Date, store and department for future processing
df_by_store = df_all.groupby(["Date", "Store","Dept"], as_index=False).agg({                                       
                                         'Fuel_Price': 'mean',
                                         'CPI': 'mean',
                                         'Unemployment': 'mean', 
                                         'Weekly_Sales': 'sum',
                                          'IsHoliday': 'mean'
                                           })


df_by_store.Date = pd.to_datetime(df_by_store.Date, errors='coerce')
df_by_store.set_index('Date', inplace=True)

##

## department

###### Train, test and prediction 
# Train and test on 2012 data to determine accuracy


from statsmodels.tsa.holtwinters import ExponentialSmoothing

##########################################
#prediction on all data
fit_model = ExponentialSmoothing(df_by_date_new['Weekly_Sales'][:-2],
                                 trend = 'add',
                                 seasonal = 'add',
                                 seasonal_periods = 52).fit()

future_prediction_all = fit_model.forecast(8)
future_prediction_all


print(future_prediction_all.head(8))

plt.figure(figsize=(20, 10))
plt.plot(df_by_date_new.index, df_by_date_new.Weekly_Sales)
plt.plot(future_prediction_all, '--')
plt.legend(['2010-2012 actual', '2013 forecast'])

 ## Validation using MAPE

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("Mean Absolute Percentage Error for all = {a}%".format(a=mean_absolute_percentage_error(df_by_date_new.Weekly_Sales[120:],future_prediction_all)))


##########################################################
# Prediction on Store level, first 4 Stores
for x in [1,2,3,4]:
    df_by_store_l = df_by_store[(df_by_store.Store == x) & (df_by_store.Dept == 1)].sort_values('Date')
    
    df_by_store_new = df_by_store_l.resample('W').mean().fillna(method='bfill') #extrapolate
    multi_plot = seasonal_decompose(df_by_store_new['Weekly_Sales'], model = 'add', extrapolate_trend='freq')
    print(df_by_store_new)

   
    #Fitting
    fit_model_s = ExponentialSmoothing(df_by_store_new['Weekly_Sales'][:-2],
                                 trend = 'add',
                                 seasonal = 'add',
                                 seasonal_periods = 52).fit()

    #We make future prediction for 8 weeks
    future_prediction = fit_model_s.forecast(8)
    future_prediction
    
    #future_prediction = future_prediction.sort_index(ascending=False)
    
    #print(future_prediction.head(20))
    
    
    # Plot
    plt.figure(figsize=(20, 10))
    plt.plot(df_by_store_new.index, df_by_store_new.Weekly_Sales)
    plt.legend(['2010-2012 actual', '2013 forecast'])
    
    
    ## Validation using MAPE 
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("Mean Absolute Percentage Error for store = {a}%".format(a=mean_absolute_percentage_error(df_by_store_new.Weekly_Sales[120:],future_prediction)))

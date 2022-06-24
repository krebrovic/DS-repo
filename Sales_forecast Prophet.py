import pandas as pd
from fbprophet import Prophet
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
from pandas import to_datetime
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import plotly.graph_objects as go
#from prophet.plot import plot_plotly, plot_components_plotly


# load data

df_stores = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/stores_dataset.csv')
df_features = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/Features_dataset.csv', parse_dates = ['Date'])
df_sales = pd.read_csv('C:/Users/krebrovic/Desktop/Zadatak/_data/sales_dataset.csv', parse_dates = ['Date'])

# clean data
df_features = df_features[df_features.Date.dt.date <= df_sales.Date.dt.date.max()]

# merge data
df_all_1 = df_features.merge(df_sales, 'right', on = ['Date', 'Store', 'IsHoliday'])

df_all = df_all_1.merge(df_stores, 'left', on = 'Store')
print('New Min date in features ds', df_all.Date.dt.date.min(), ',max: ', df_all.Date.dt.date.max())

# make data "Prophet compatible", Prophet needs data in exact format
df_all = df_all[['Date', 'Weekly_Sales']]
df_all.columns = ['ds', 'y']

#######################################
# Predicting 2013 sales


# loop through stores
for x in [1,2,3,4]:

    store = df_all[(df_all.Store == x) & (df_all.Dept == 1)].sort_values('Date')
    
    
    df_all_p = store[['Date', 'Weekly_Sales']]

    df_all_p.columns = ['ds', 'y']


    model = Prophet(interval_width=0.95,daily_seasonality=False, 
            weekly_seasonality=True)
# Fitting
    model.fit(df_all_p)

    future= model.make_future_dataframe(periods=8,freq='W')
#future.columns = ['ds']
#future['ds']= to_datetime(future['ds'])

# Prediction
    forecast = model.predict(future)

#show data
    forecast.sort_values(by='ds', ascending = False, inplace=True)
    print('Forecast for :' +str(x))
    print(forecast.head(10))

# print Plot
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Plot
    model.plot(forecast)
    pyplot.show()
    
    
    #data validation, using MAPE, horizon
    #  the initial training period is set to three times the horizon, and cutoffs are made every half a horizon
df_cv = cross_validation(model,horizon='56 days')


df_p = performance_metrics(df_cv)
print(df_p.head())




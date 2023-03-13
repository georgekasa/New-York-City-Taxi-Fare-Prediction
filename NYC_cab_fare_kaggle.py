import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cudf
import cupy as cp
import matplotlib.pyplot as plt
from cuml.preprocessing import StandardScaler, OneHotEncoder
from cuml.pipeline import Pipeline
from cuml.model_selection import train_test_split
from cuml.compose import ColumnTransformer


def histogram_plot(df, column, bins, title, x_label, y_label, color):
    map = plt.hist(df[column], bins=21, range=(0,20), ec='white')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def scatter_plot(y, title = "something", x_label = "values", y_label = None, color = "blue"):
    plt.scatter(np.arange(len(y)), y, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def createTimeFeatures(df):
    # Convert pickup_datetime to datetime 2009-06-15 17:26:21 UTC
    dt = cudf.to_datetime(df['pickup_datetime'])
    year = df['pickup_datetime'].dt.year
    month = df['pickup_datetime'].dt.month# 1-12
    date = df['pickup_datetime'].dt.dayofweek# 0-6 oti nane to allo 1-12 auto apo 0-6
    hour = df['pickup_datetime'].dt.hour
    min =  df['pickup_datetime'].dt.minute
    #add new columns
    df['year'] = year
    df["month"] = month
    df["date"] = date
    df["hour"] = hour
    df["minute"] = min
    createRushingHourFeatures(df, "hour")
    weekend(df, "date")
    coldMonths(df, "month")
    warmMonths(df, "month")
    nightCharge(df, "hour")
    

def createRushingHourFeatures(df, columnHour):
    df["rush_hour"] = df[columnHour].isin([7, 8, 9, 16, 17, 18])
    
def weekend(df, dayofweek):
    df["weekend"] = df[dayofweek].isin([5, 6])

def coldMonths(df, month):
    df["coldMonth"] = df[month].isin([1, 2, 3, 10, 11, 12])

def warmMonths(df, month):
    df["warmMonth"] = df[month].isin([4, 5, 6, 7, 8, 9])
    
def nightCharge(df, hour):
    df["nightCharge"] = df[hour].isin([0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23])
    
    

def distance(df,pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    #eucledian distance
    df["distance"] = cp.sqrt((pickup_long - dropoff_long)**2 + (pickup_lat - dropoff_lat)**2)
    
    
    
start = time.time()
df1 = cudf.read_csv('/media/gkasap/ssd256gb/datasets/new-york-city-taxi-fare-prediction/train.csv', nrows = 1_500_000, parse_dates=["pickup_datetime"])
# 4 seconds with pandas
# 1.39 seconds with cudf!
df2 = cudf.read_csv('/media/gkasap/ssd256gb/datasets/new-york-city-taxi-fare-prediction/test.csv', parse_dates=["pickup_datetime"])
df = cudf.concat([df1, df2])
#shuffle them and reset index
df = df.sample(frac=1, random_state = 32).reset_index(drop=True)
print('Time to read csv: ', time.time() - start)


#pick 10 random rows
#print(df.sample(10))
#Index(['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude',
#       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
 #      'passenger_count'],
 
 
 ############################# 1. Data Exploration #############################
#fare mount min is negative? and maximum is 1.27k?
# 3M - 2.999977M = 23 rows have no dropoff coordinates
# they are nan there 23 rows let us check them
#df[df['dropoff_longitude'].isnull()] random nothing correlated so just drop them also passenger count is 0
df.dropna(inplace=True)

#also for longtiude pick we have mean -72 and minmum value -3.42k!! and max 3.43k and s^2 is = 13.2

# Choose cab rides whose pickup and dropoff are the US Mainland
# Declare constants
latmin = 5.496100
latmax = 71.538800
longmin = -124.482003
longmax = -66.885417
old_rows = df.shape[0]
#based on the above values we can filter the data
df = df[(df['pickup_longitude'] > longmin) & (df['pickup_longitude'] < longmax) & (df['pickup_latitude'] > latmin) & (df['pickup_latitude'] < latmax) & (df['dropoff_longitude'] > longmin) & (df['dropoff_longitude'] < longmax) & (df['dropoff_latitude'] > latmin) & (df['dropoff_latitude'] < latmax)]#github copilot thank you

#remove passenger count = 0 and passenger count is 9 which could a mistake or van ?? I can check it fare amount and later (or if they have the same pickup longtiude e.g. an airport or train station)
#only 1 row has passenger count = 9 so is an error so we can drop it
df = df[df['passenger_count'] > 0 & (df['passenger_count'] < 9)]

#remove negative fare amount and 
df = df[df['fare_amount'] > 0]


print("number of rows dropped:  or like percentage", -df.shape[0]+  old_rows, cp.round(100*(old_rows - df.shape[0])/old_rows, 3))

#histogram_plot(df.to_pandas(), 'fare_amount', 8, 'fare amount', 'Count', 'fare amount', 'blue')
createTimeFeatures(df)
#calculate euclidean distance between pickup and dropoff
distance(df, df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])

#convert month and date to categorical e.g. september, october, november, december
month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df["month"] = df["month"].map(month_dict)

weekday_dict = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df["date"] = df["date"].map(weekday_dict)
df["date"].unique()
df_bckup = df.copy()
df = df.drop(['pickup_datetime', "key", "hour", "minute"], axis=1)
df.reset_index(drop=True, inplace=True)
##############################
#finish the data exploration
##############################


#to do in  2023/03/10
#start ML pipeline
#check linear regression, ridge, xgboost & neural network

######pipeline start########
numeric_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'year', 'distance']
discrete_cat_features = ['passenger_count', 'date', 'month']

#transform binary categorical features to 0 and 1
binary_cat_features = ['rush_hour', 'weekend', 'coldMonth', 'warmMonth', 'nightCharge']
df[binary_cat_features] = df[binary_cat_features].astype("float32")

discrete_cat_Pipeline = Pipeline([('discrete', OneHotEncoder(handle_unknown='error'))])#add drop parameter 
numericPipeline = Pipeline([('scaler', StandardScaler())])


transformer = ColumnTransformer([("numeric_preprocessing", numericPipeline, numeric_features)], remainder='passthrough')
transformerCategorical = ColumnTransformer([("categorical_preprocessing", discrete_cat_Pipeline, discrete_cat_features)], remainder='passthrough')

print("Finish ")
#https://github.com/coreyjwade/NYC_Cab_Fare/blob/master/NYC_Pipeline_Tests.ipynb
#https://github.com/coreyjwade/NYC_Cab_Fare/blob/master/NYC_Data_Wrangling.ipynb

##############################################
#####3. ML pipeline###########################

x_train, x_test, y_train, y_test = train_test_split(df.drop("fare_amount", axis = 1), df["fare_amount"], test_size=0.2, random_state=42)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#Numeric features
transformer.fit(x_train[numeric_features])
x_train_numeric_scaled = transformer.transform(x_train[numeric_features])
x_test_numeric_scaled = transformer.transform(x_test[numeric_features])
x_train_numeric_scaled.rename(columns={0: 'pickup_longitude', 1: 'pickup_latitude', 2: 'dropoff_longitude', 3: 'dropoff_latitude', 4: 'year', 5: 'distance'}, inplace=True)
x_test_numeric_scaled.rename(columns={0: 'pickup_longitude', 1: 'pickup_latitude', 2: 'dropoff_longitude', 3: 'dropoff_latitude', 4: 'year', 5: 'distance'}, inplace=True)
# target'fare_amount', 
#numeric features = [ pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', "year", "distance"] = 6
#binary_cat_features = ['rush_hour', 'weekend', 'coldMonth', 'warmMonth', 'nightCharge'] = 5
#discrete_cat_features = ['passenger_count', 'date', 'month'] = 3
#total = 6 + 5 + 3 + 1 (target)= 15


#categoical features = 8
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(x_train[discrete_cat_features])
x_train_encoded = enc.transform(x_train[discrete_cat_features])
x_test_encoded = enc.transform(x_test[discrete_cat_features])
x_train_encoded_df = cudf.DataFrame(x_train_encoded, columns=enc.get_feature_names(), dtype="float32")
x_test_encoded_df = cudf.DataFrame(x_test_encoded, columns=enc.get_feature_names(), dtype="float32")




#merge all features together, numeric, categorical and binary 
#/’index’, 1/’columns’}, default 0
x_train_all = cudf.concat([x_train_numeric_scaled, x_train_encoded_df, x_train[binary_cat_features]], axis=1)
x_test_all = cudf.concat([x_test_numeric_scaled, x_test_encoded_df, x_test[binary_cat_features]], axis=1)
import random
from cuml import Ridge
from cuml.linear_model import Ridge
from cuml.metrics import mean_squared_error

random_rows = x_test_all.sample(n=20, random_state=69)

alpha = cp.array([1])
ridge = Ridge(alpha=alpha, fit_intercept=True, normalize=False,
              solver='svd')

result_ridge = ridge.fit(x_train_all, y_train)
y_pred_random = result_ridge.predict(random_rows)
y_test_random = y_test.iloc[random_rows.index]


y_pred = ridge.predict(x_test_all)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error of Ridge:", mse)

# MSE with ridge 82

#scatter_plot(df.to_pandas(), "distance")





##############################################
#########xg boost#############################
##############################################

import matplotlib.pyplot as plt
import xgboost as xgb
import cupy as cp
#sum(negative instances) / sum(positive instances)
params = {
    'max_depth':    8,
    'max_leaves':   2**8,
    'tree_method':  'gpu_hist',
    'objective':    'reg:squarederror',
    'grow_policy':  'depthwise',
    'eval_metric':  ['rmse'],
    'subsample':    '0.8',
    'seed':         42,
    'eta':          '0.1',
}
dtrain = xgb.DMatrix(x_train_all, y_train)
model = xgb.train(params, dtrain, num_boost_round=100)
dtest = xgb.DMatrix(x_test_all)
y_pred_xgboost = model.predict(dtest)

mse_xgboost = mean_squared_error(y_test, y_pred)
print("Mean squared error of XGBoost:", mse_xgboost)
#cudf.concat([x_train_numeric_scaled, x_train_encoded_df, x_train[binary_cat_features]], axis=1)










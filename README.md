# New-York-City-Taxi-Fare-Prediction
kaggle New York City Taxi Fare Prediction


this is from https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview
It has wc -l train.csv  -> 55_423_856 train.csv
#I could not train with my single gpu on 55M rows the pre proccessing to be faster was made again in
cudf, cupy, etc I achieved to trained with 1.5M rows with XGboost and SGD

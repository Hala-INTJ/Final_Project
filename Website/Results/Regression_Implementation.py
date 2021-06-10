# Import dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import json

import datetime
import re  

from sqlalchemy import create_engine

# Supress warnings
import warnings
warnings.filterwarnings("ignore")


def run_model(X,y,modelName,model):
   
    # Creating training and testing subsets
    split = int(X.shape[0]*0.7)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
              
    # Standarize the data
    X_scaler = StandardScaler().fit(X_train)
    Y_scaler = StandardScaler().fit(pd.DataFrame(y_train))

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = Y_scaler.transform(pd.DataFrame(y_train))
    y_test_scaled = Y_scaler.transform(pd.DataFrame(y_test))
    
    # Fit and evaluate each model
    model.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = model.predict(X_test_scaled)

    y_test = Y_scaler.inverse_transform(pd.DataFrame(y_test_scaled))
    y_pred = Y_scaler.inverse_transform(pd.DataFrame(y_pred_scaled))

    y_test = pd.Series(pd.DataFrame.from_records(y_test)[0].values)
    y_pred = pd.Series(pd.DataFrame.from_records(y_pred)[0].values)
        
    results = pd.DataFrame({
    "Prediction": y_pred, 
    "Actual": y_test
    }).reset_index(drop=True)

    # if modelName in ['Linear','SVR']:
    #     imp = model.coef_[0]
    #     importances, features = zip(*sorted(zip(model.coef_[0],list(X.columns))))
    #     importances = pd.DataFrame(sorted(zip(model.coef_[0], list(X.columns))))
    # else:
    #     importances = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns), reverse=True))
    #     importances.columns = ['value', 'feature_name']
    #     features = importances["feature_name"]
    #     importances = importances.set_index('feature_name')

    adjusted_r2 = 1-(1-(r2_score(y_test, y_pred)))*((len(X_test_scaled)-1))/(len(X_test_scaled)-len(X_test_scaled[0])-1)

    d={}
    d['Name'] = modelName
    d['R2'] = (f"{r2_score(y_test, y_pred):.4f}")
    d['Adjusted R2'] = (f"{adjusted_r2:.4f}")
    d['Mean Square Error'] = (f"{mean_squared_error(y_test, y_pred):.4f}")
    d['Root Mean Square Error'] = (f"{math.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    d['Mean Absolute Error'] = (f"{mean_absolute_error(y_test, y_pred):.4f}")
    d['Predicted Correctly'] = (f"{((y_pred > 0.35) & (y_test > 0.35)).sum()}")
    d['Actual > 0.35'] = (f"{(y_test > 0.35).sum()}")

    return d, results.to_dict()

models = {
    'Linear': LinearRegression(),
    'SVR': SVR(kernel='linear'),
    'DecisionTree': tree.DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor()}

trains = {
    "9": {
        "df_1": ['Time', 'T5-S3-PRE-FeCL2'],
        "df_2": r'(^T5.*-P9-.*|Time)',
        "df_3": r'(^T5.*-P9.*|Time)',
        "df_4": r'(^T5.*-S17.*|Time)',
        "target": "T5-S3-SEC-S17-TP"
    },
    "10": {
        "df_1": ['Time', 'T5-S3-PRE-FeCL2'],
        "df_2": r'(^T5.*-P10-.*|Time)',
        "df_3": r'(^T5.*-P10.*|Time)',
        "df_4": r'(^T5.*-S18.*|Time)',
        "target": "T5-S3-SEC-S18-TP"
    },
    "11": {
        "df_1": ['Time', 'T5-S3-PRE-FeCL2'],
        "df_2": r'(^T5.*-P11-.*|Time)',
        "df_3": r'(^T5.*-P11.*|Time)',
        "df_4": r'(^T5.*-S19.*|Time)',
        "target": "T5-S3-SEC-S19-TP"
    },
    "12": {
        "df_1": ['Time', 'T6-S3-PRE-FeCL2'],
        "df_2": r'(^T6.*-P12-.*|Time)',
        "df_3": r'(^T6.*-P12.*|Time)',
        "df_4": r'(^T[5,6].*-S20.*|Time)',
        "target": "T5-S3-SEC-S20-TP"
    },
    "13": {
         "df_1": ['Time', 'T6-S3-PRE-FeCL2'],
        "df_2": r'(^T6.*-P13-.*|Time)',
        "df_3": r'(^T6.*-P13.*|Time)',
        "df_4": r'(^T[5,6].*-S21.*|Time)',
        "target": "T5-S3-SEC-S21-TP"
    },
    "14": {
        "df_1": ['Time', 'T6-S3-PRE-FeCL2'],
        "df_2": r'(^T6.*-P14-.*|Time)',
        "df_3": r'(^T6.*-P14.*|Time)',
        "df_4": r'(^T[5,6].*-S22.*|Time)',
        "target": "T5-S3-SEC-S22-TP"
    }        
}    

for modelName, model in models.items():

    model_results = {}

    for train, vars in trains.items():
        # Connecting to the Database
        engine = create_engine("postgresql://postgres:postgres@localhost/WWTP")
        conn = engine.connect()

        # Reading SQL query into a Dataframe 
        df_1 = pd.read_sql_query('select * from "Preliminary"', con=conn)
        df_2 = pd.read_sql_query('select * from "Primary"', con=conn)
        df_3 = pd.read_sql_query('select * from "Aeration"', con=conn)
        df_4 = pd.read_sql_query('select * from "Secondary"', con=conn)

        # Close the connection
        conn.close()

        df_1 = df_1[vars['df_1']] 
        specific_columns = []
        for col in list(df_2.columns):
            if (re.match(vars['df_2'], col)):
                specific_columns.append(col)         
        df_2 = df_2[specific_columns]
        specific_columns = []
        for col in list(df_3.columns):
            if (re.match(vars['df_3'], col)): 
                specific_columns.append(col)
        df_3 = df_3[specific_columns]
        specific_columns = []
        for col in list(df_4.columns):
            if (re.match(vars['df_4'], col)):
                specific_columns.append(col)
        df_4 = df_4[specific_columns]   

        # Merging Dataframes
        df_temp_1 = pd.merge(df_1, df_2, on='Time', how='outer')
        df_temp_2 = pd.merge(df_temp_1, df_3, on='Time', how='outer')
        df = pd.merge(df_temp_2, df_4, on='Time', how='outer')

        # Keeping the records satring on July 1st, 2017
        df = df[df['Time'] >= datetime.datetime(2017,7,1)].sort_values(by='Time')

        # Resetting the index
        df.reset_index(inplace=True, drop=True)

        # Dropping columns due to missing data until November 2018
        specific_columns = []
        for col in df.columns:
            if (re.match(r'(^.*-PRI-.*-TKN|^.*-PRI-.*-Ammonia|^.*-PRI-.*-Nitrate|^.*-PRI-.*-Nitrite)', col)):
                specific_columns.append(col)
        df.drop(columns=specific_columns, inplace = True, axis = 1)            

        # Dropping NaN
        df = df.dropna()

        # Covert Time into numerical columns
        df['month'] = df['Time'].dt.month
        df['week'] = df['Time'].dt.week
        df['day'] = df['Time'].dt.day

        # Create a Series for "Time" column
        time_column = df["Time"]

        # Drop the time, year and month columns
        df.drop(['Time'], inplace = True, axis = 1)

        y = df[vars['target']]
        X = df.drop(columns=vars['target'])

        outcome, results = run_model(X,y,modelName,model)

        model_results[train] = {
            "outcome": outcome,
            "results": results
        }

    fileName = modelName + "_Regression.json"
    with open(fileName, 'w', encoding='utf-8') as outfile:
        json.dump(model_results, outfile, indent=4)
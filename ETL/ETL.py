
# Import dependencies
from statsmodels.tsa.seasonal import STL
from scipy import stats
import pandas as pd
import numpy as np

def find_outliers(df, columns_list, time_column): 

    results = []

    for col in columns_list:
        timeseriesdemo_agg = df.groupby(['Time'])[col].sum().reset_index()
        timeseriesdemo_agg = timeseriesdemo_agg[['Time', col]].set_index('Time')
        timeseriesdemo_agg


        score_threshold = 7

        # STL method does Season-Trend decomposition using LOESS
        stl = STL(timeseriesdemo_agg)
        # Fitting the data - Estimate season, trend and residuals components.
        res = stl.fit()

        output = timeseriesdemo_agg.copy()
        # Create dataframe columns from decomposition results
        output['residual']= res.resid
        output['trend']= res.trend
        output['seasonal'] = res.seasonal
        output['weights'] = res.weights

        # Baseline is generally seasonal + trend 
        output['baseline'] = output['seasonal']+output['trend']
        
        # Calculate zscore based on residual column - this column does not contain seasonal/trend components
        output['score'] = stats.zscore(output['residual'])
            
        # Create positive and negative columns based on threshold(3) and seasonal components
        output.loc[(output['score'] > score_threshold) & (output['seasonal'] > 0),'anomalies'] = 1
        output.loc[(output['score'] > score_threshold) & (output['seasonal'] < 0),'anomalies'] = -1
        output.loc[(output['score'] < score_threshold),'anomalies'] = 0
        
        # Resetting Index
        output = output.reset_index()

        # Filter the dataframe and displaying only Flagged anomalies
        anomaly_df = output[output['anomalies']==1]
        anomaly_tagname = anomaly_df.columns[1]
        for index, row in anomaly_df.iterrows():
            anomaly_index = index
            anomaly_value = row[col]
            anomaly_score = row['score']
            
            d={}
            d['tag_name'] = anomaly_tagname
            d['value'] =  anomaly_value
            d['index'] = anomaly_index
            d['score'] = anomaly_score

            results.append(d)   
 
    return pd.DataFrame(results)


def clean_outliers(df, columns_list, time_column):  
    outliers_df = find_outliers(df, columns_list, time_column)

    # Iterate through the dataframe
    for index, row in outliers_df.iterrows():
        df.loc[row['index'],row['tag_name']] = np.nan

    return df

def find_outliers_analytes(df, columns_list, time_column): 

    results = []

    for col in columns_list:
        timeseriesdemo_agg = df.groupby(['Time'])[col].sum().reset_index()
        timeseriesdemo_agg = timeseriesdemo_agg[['Time', col]].set_index('Time')
        timeseriesdemo_agg


        score_threshold = 15

        # STL method does Season-Trend decomposition using LOESS
        stl = STL(timeseriesdemo_agg)
        # Fitting the data - Estimate season, trend and residuals components.
        res = stl.fit()

        output = timeseriesdemo_agg.copy()
        # Create dataframe columns from decomposition results
        output['residual']= res.resid
        output['trend']= res.trend
        output['seasonal'] = res.seasonal
        output['weights'] = res.weights

        # Baseline is generally seasonal + trend 
        output['baseline'] = output['seasonal']+output['trend']
        
        # Calculate zscore based on residual column - this column does not contain seasonal/trend components
        output['score'] = stats.zscore(output['residual'])
            
        # Create positive and negative columns based on threshold(3) and seasonal components
        output.loc[(output['score'] > score_threshold) & (output['seasonal'] > 0),'anomalies'] = 1
        output.loc[(output['score'] > score_threshold) & (output['seasonal'] < 0),'anomalies'] = -1
        output.loc[(output['score'] < score_threshold),'anomalies'] = 0
        
        # Resetting Index
        output = output.reset_index()

        # Filter the dataframe and displaying only Flagged anomalies
        anomaly_df = output[output['anomalies']==1]
        anomaly_tagname = anomaly_df.columns[1]
        for index, row in anomaly_df.iterrows():
            anomaly_index = index
            anomaly_value = row[col]
            anomaly_score = row['score']
            
            d={}
            d['tag_name'] = anomaly_tagname
            d['value'] =  anomaly_value
            d['index'] = anomaly_index
            d['score'] = anomaly_score

            results.append(d)   
 
    return pd.DataFrame(results)

def clean_outliers_analytes(df, columns_list, time_column):  
    outliers_df = find_outliers_analytes(df, columns_list, time_column)

    # Iterate through the dataframe
    for index, row in outliers_df.iterrows():
        df.loc[row['index'],row['tag_name']] = np.nan

    return df    





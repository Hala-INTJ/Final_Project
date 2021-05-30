# Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_models(X,y):
    # Chosen Machine Learning models
    models = {'Linear Regression': LinearRegression(),
              'SVR (Linear)': SVR(kernel='linear'),
              'Descision Tree Regression': tree.DecisionTreeRegressor(),
              'Random Forest Regression': RandomForestRegressor(),
              'Gradient Boosting Regressor': GradientBoostingRegressor(),
              'Ada Boost Regressor': AdaBoostRegressor()}

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
    outcome = []
    for name,model in models.items():
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
        
        fig,(ax, ax2) = plt.subplots(2,1,figsize=(10,12))

        results.plot(kind='line',y='Prediction',ax=ax)
        results.plot(kind='line',y='Actual', color='red', ax=ax)
        ax.set_title(name)

        if name in ['Linear Regression','SVR (Linear)']:
            imp = model.coef_[0]
            imp, features = zip(*sorted(zip(model.coef_[0],list(X.columns))))
            ax2.barh(range(len(features)), imp, align='center',color='purple')
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_title(name)
        else:
            importances = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns), reverse=True))
            importances.columns = ['value', 'feature_name']
            importances = importances.set_index('feature_name')
            (importances[:10]).plot.barh(color='purple', ax=ax2) 
            ax2.legend_= None
            ax2.invert_yaxis()
            ax2.set_title(name)

        plt.show()

        adjusted_r2 = 1-(1-(r2_score(y_test, y_pred)))*((len(X_test_scaled)-1))/(len(X_test_scaled)-len(X_test_scaled[0])-1)

        d={}
        d['Name'] = name
        d['R2'] = (f"{r2_score(y_test, y_pred):.4f}")
        d['Adjusted R2'] = (f"{adjusted_r2:.4f}")
        d['Mean Square Error'] = (f"{mean_squared_error(y_test, y_pred):.4f}")
        d['Root Mean Square Error'] = (f"{math.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        d['Mean Absolute Error'] = (f"{mean_absolute_error(y_test, y_pred):.4f}")
        d['Predicted Correctly'] = (f"{((y_pred > 0.35) & (y_test > 0.35)).sum()}")
        d['Actual > 0.35'] = (f"{(y_test > 0.35).sum()}")
        outcome.append(d)
    return pd.DataFrame(outcome)

def neural_model(X, y, layer_nodes, activation, epochs):
    
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
    
    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
    model = Sequential()
    for index, nodes in enumerate(layer_nodes):
        if index == 0:
            model.add(Dense(units=nodes, input_dim=len(X_train_scaled[0]), activation=activation))
        else:
            model.add(Dense(units=nodes, activation=activation))
    model.add(Dense(units=1, activation='linear'))

    # Check the structure of the model
    print(model.summary())
    
    # Compile the model
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse','mae'])
    
    model.fit(X_train_scaled, y_train_scaled, epochs=500,verbose=0)
    y_pred_scaled = model.predict(X_test_scaled)

    y_test = Y_scaler.inverse_transform(pd.DataFrame(y_test_scaled))
    y_pred = Y_scaler.inverse_transform(pd.DataFrame(y_pred_scaled))

    y_test = pd.Series(pd.DataFrame.from_records(y_test)[0].values)
    y_pred = pd.Series(pd.DataFrame.from_records(y_pred)[0].values)
        
    results = pd.DataFrame({
        "Prediction": y_pred, 
        "Actual": y_test
        }).reset_index(drop=True)
        
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    results.plot(kind='line',y='Prediction',ax=ax)
    results.plot(kind='line',y='Actual', color='red', ax=ax)
    plt.show()

    adjusted_r2 = 1-(1-(r2_score(y_test, y_pred)))*((len(X_test_scaled)-1))/(len(X_test_scaled)-len(X_test_scaled[0])-1)
        
    d={}
    d['Name'] = (f"Neural Model: {layer_nodes}, {activation}, {epochs}")
    d['R2'] = (f"{r2_score(y_test, y_pred):.4f}")
    d['Adjusted R2'] = (f"{adjusted_r2:.4f}")
    d['Mean Square Error'] = (f"{mean_squared_error(y_test, y_pred):.4f}")
    d['Root Mean Square Error'] = (f"{math.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    d['Mean Absolute Error'] = (f"{mean_absolute_error(y_test, y_pred):.4f}")
    d['Predicted Correctly'] = (f"{((y_pred > 0.35) & (y_test > 0.35)).sum()}")
    d['Actual > 0.35'] = (f"{(y_test > 0.35).sum()}")
    return pd.DataFrame([d])     
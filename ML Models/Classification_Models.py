# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report
from imblearn.combine import SMOTEENN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_models(X,y,smoteenn=False):
    # Chosen Machine Learning models
    models = {'Logistic Regression (lbfgs)': LogisticRegression(solver='lbfgs'),
              'SVC (poly)': SVC(kernel='poly'),
              'Descision Tree Classification': DecisionTreeClassifier(),
              'Random Forest Classification': RandomForestClassifier(),
              'Balanced Random Forest Classification': BalancedRandomForestClassifier(),
              'Easy Ensemble Classification': EasyEnsembleClassifier(),
              'Gradient Boosting Classifier': GradientBoostingClassifier(),
              'AdaBoostClassifier': AdaBoostClassifier()}

    # Creating training and testing subsets
    split = int(X.shape[0]*0.7)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    if smoteenn:
        smote_enn = SMOTEENN()
        X_train, y_train = smote_enn.fit_resample(X_train, y_train)

              
    # Standarize the data
    X_scaler = StandardScaler().fit(X_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Fit and evaluate each model
    outcome = []
    for name,model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        print(f"{name}\n")

        if name in ['Logistic Regression (lbfgs)']:
            imp = model.coef_[0]
            imp, features = zip(*sorted(zip(model.coef_[0],list(X.columns))))
            fig,ax = plt.subplots(figsize=(10,6))
            ax.barh(range(len(features)), imp, align='center',color='purple')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_title(name)
            plt.show()
        elif name in ['Descision Tree Classification','Random Forest Classification','Balanced Random Forest Classification']:
            importances = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns), reverse=True))
            importances.columns = ['value', 'feature_name']
            importances = importances.set_index('feature_name')
            fig,ax = plt.subplots(figsize=(10,6))
            (importances[:10]).plot.barh(color='purple', ax=ax) 
            ax.invert_yaxis()
            ax.legend_= None
            ax.set_title(name)
            plt.show()
        else: 
            pass

        print(f"Confusion matrix:\n {confusion_matrix(y_test, y_pred)}\n")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n\n")
        print(f"----------------------------------------------")
        
        CM = confusion_matrix(y_test, y_pred) 

        d={}
        d['Name'] = (f"{name}{' (SMOTEENN)' if smoteenn else ''}")
        d['Accuracy Score'] = (f"{accuracy_score(y_test, y_pred):.4f}")
        d['Balanced Accuracy Score'] = (f"{balanced_accuracy_score(y_test, y_pred):.4f}")
        d['"Exceedance" Predicted Correctly (TP)'] = CM[0][0]
        d['"Exceedance" Predicted Incorrectly (FP)'] = CM[0][1]
        d['"No Exceedance" Predicted Incorrectly (FN)'] = CM[1][0]
        d['"No Exceedance" Predicted Correctly (TN)'] = CM[1][1]
        d['Actual'] = (f"{(y_test == 0).sum()}")
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

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
    model = Sequential()
    for index, nodes in enumerate(layer_nodes):
        if index == 0:
            model.add(Dense(units=nodes, input_dim=len(X_train_scaled[0]), activation=activation))
        else:
            model.add(Dense(units=nodes, activation=activation))
    model.add(Dense(units=1, activation='sigmoid'))

    # Check the structure of the model
    print(model.summary())
    
    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(X_train_scaled, y_train, epochs=500,verbose=0)
    y_pred = model.predict(X_test_scaled)

    model_loss, model_accuracy = model.evaluate(X_test_scaled,y_test)
        
    d={}
    d['Name'] = (f"Neural Model: {layer_nodes}, {activation}, {epochs}")
    d['Accuracy Score'] = (f"{model_accuracy:.4f}")
    d['Model Loss'] = (f"{model_loss:.4f}")
    d['Predicition'] = (f"{(y_pred == 0).sum()}")
    d['Actual'] = (f"{(y_test == 0).sum()}")
    return pd.DataFrame([d])     
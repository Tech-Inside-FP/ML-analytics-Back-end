# imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

import base64

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io



def simple_linear_regression(train_data, test_data):

    # Extrai caracteristicas dos dados de treinamento e teste
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    #modelo
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #metricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    #plot regressão
    plt.scatter(X_train, y_train)
    plt.plot(X_test, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regressão Linear Simples')
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()
    image_bytes = base64.b64encode(s=image_bytes.getvalue())

    return mae, mse, rmse, r2, image_bytes, model


"""###Mult Linear Regression"""

def multiple_linear_regression(train_data, test_data):

    # Extrai caracteristicas dos dados de treinamento e teste
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    #modelo
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #metricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2, np.asarray(a=y_pred), np.asarray(a=y_test)


"""### Model Bayes"""

def naive_bayes_classifier(train_data, test_data):
    # Extrai caracteristicas dos dados de treinamento e teste
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    # Treina modelo Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    #previsao dados de teste
    y_pred = nb.predict(X_test)

    #metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1, np.asarray(a=y_pred), np.asarray(a=y_test)


"""### Modelo de Decisão Arvore"""

def decision_tree_classifier(train_data, test_data):
    #Extrai caracteristicas dos dados de treinamento e teste
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    #treina modelo arvore de decisao
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    #previsao dados teste
    y_pred = dtc.predict(X_test)

    #metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1, np.asarray(a=y_pred), np.asarray(a=y_test)

"""### Model KNN"""

def knn_classifier(train_data, test_data):
    #Extrai caracteristicas dados de treinamento e teste
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    #cria modelo KNN
    best_acc = 0
    best_pre = 0
    best_rec = 0
    best_f1 = 0
    best_pred = None
    for k in range(1,31):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        #previsao dados de teste
        y_pred = knn.predict(X_test)

        #metricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if accuracy > best_acc:
            best_acc = accuracy
            best_pre = precision
            best_rec = recall
            best_f1 = f1
            best_pred = y_pred

    return best_acc, best_pre, best_rec, best_f1, np.asarray(a=best_pred), np.asarray(a=y_test)

def train(model_type: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:

    models = {
        'regression': [('multi linear', multiple_linear_regression)],
        'classification': [('knn', knn_classifier), ('decision tree', decision_tree_classifier), ('naive bayes', naive_bayes_classifier)]
    }
    metrics_names = {
        'regression': ['mae', 'mse', 'rmse', 'r2'],
        'classification': ['accuracy', 'precision', 'recall', 'f1', 'cm']
    }
    metrics = metrics_names[model_type] + ['yhat', 'y']
    results = {}
    for name, model in models[model_type]:

        results[name] = {}

        output = model(train, test)

        for met_name, value in zip(metrics, output):
            results[name][met_name] = value

    return results
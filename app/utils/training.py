# imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io



def simple_linear_regression(x, y):
    #modelo
    model = LinearRegression()

    x = x.reshape(-1, 1)
    model.fit(x, y)

    y_pred = model.predict(x)

    #metricas
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    #plot regressão
    plt.scatter(x, y)
    plt.plot(x, y_pred, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regressão Linear Simples')
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()

    return mae, mse, rmse, r2, model, image_bytes

# dados de exemplo
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 6, 8])

#chama funcao simple_linear_regression
simple_linear_regression(x, y)

"""###Mult Linear Regression"""

def multiple_linear_regression(x, y):
    #modelo
    model = LinearRegression()

    model.fit(x, y)

    y_pred = model.predict(x)

    #metricas
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    #plot metricas
    metrics = ['MAE', 'MSE', 'RMSE', 'R²']
    values = [mae, mse, rmse, r2]
    plt.bar(metrics, values)
    plt.title('Métricas de Regressão Linear Múltipla')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()

    return mae, mse, rmse, r2, model, image_bytes

#dados de exemplo
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = np.array([3, 5, 7, 9, 11])

#chama funcao multiple_linear_regression
multiple_linear_regression(x, y)

"""### Model Bayes"""

def naive_bayes_classifier(train_data, test_data):
    # Extrai caracteristicas dos dados de treinamento e teste
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['text'])
    y_train = train_data['label']
    X_test = vectorizer.transform(test_data['text'])
    y_test = test_data['label']

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

    #matriz de confusao
    cm = confusion_matrix(y_test, y_pred)

    #Plot matriz de confusao
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()

    return accuracy, precision, recall, f1, cm, nb, image_bytes

#exemplo dados de treinamento
train_data = pd.DataFrame({
    'text': [
        'este é um texto de exemplo',
        'outro exemplo de texto',
        'um terceiro exemplo de texto',
        'mais um exemplo para o modelo'
    ],
    'label': [0, 1, 0, 1]
})

#exemplo dados de teste
test_data = pd.DataFrame({
    'text': [
        'exemplo de texto para teste',
        'outro teste de modelo',
        'um terceiro teste de classificação'
    ],
    'label': [0, 1, 0]
})

#chama funcao naive_bayes_classifier
naive_bayes_classifier(train_data, test_data)

"""### Modelo de Decisão Arvore"""

def decision_tree_classifier(train_data, test_data):
    #Extrai caracteristicas dos dados de treinamento e teste
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

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
    
    #matriz de confusao
    cm = confusion_matrix(y_test, y_pred)

    #Plot arvore
    plt.figure(figsize=(12, 8))
    plot_tree(dtc, feature_names=X_train.columns, class_names=['0', '1'], filled=True, rounded=True)
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()

    return accuracy, precision, recall, f1, cm, dtc, image_bytes

"""### Model KNN"""

def knn_classifier(train_data, test_data, k):
    #Extrai caracteristicas dados de treinamento e teste
    X_train = train_data[['feature1', 'feature2', 'feature3']]
    y_train = train_data['label']
    X_test = test_data[['feature1', 'feature2', 'feature3']]
    y_test = test_data['label']

    #cria modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    #previsao dados de teste
    y_pred = knn.predict(X_test)

    #metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #matriz de confusao
    cm = confusion_matrix(y_test, y_pred)

    #Plot matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()

    return accuracy, precision, recall, f1, cm, knn, image_bytes
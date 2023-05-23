# imports
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import datetime
import re


def split_dataset(data: pd.DataFrame):

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)
    train = pd.concat(objs=[x_train, y_train], axis=1)
    test = pd.concat(objs=[x_test, y_test], axis=1)

    return train, test


def tratar_coluna_data(dado):

    filtro_data = re.compile(r'^[0-9]{4}\-[0-9]{2}\-[0-9]{2} ([0-9]{2}:[0-9]{2}:[0-9]{2})')

    if not isinstance(dado, (str)):
      dado = str(dado)
    dado = dado.strip()

    filtro = filtro_data.search(dado)
    if filtro:
      return filtro

    for fmt in ('%d-%m-%Y',
                '%d-%m-%y',
                '%d-%B-%Y',
                '%d-%B-%y',
                '%d-%b-%Y',
                '%d-%b-%y',

                '%d/%m/%Y',
                '%d/%m/%y',
                '%d/%B/%Y',
                '%d/%B/%y',
                '%d/%b/%Y',
                '%d/%b/%y',

                '%Y-%m-%d',
                '%Y-%B-%d',
                '%Y-%b-%d',
                '%y-%m-%d',
                '%y-%B-%d',
                '%y-%b-%d',

                '%Y/%m/%d',
                '%Y/%B/%d',
                '%Y/%b/%d',
                '%y/%m/%d',
                '%y/%B/%d',
                '%y/%b/%d',

                '%d de %B de %Y',
                '%d de %b de %Y',
                '%d de %B de %y',
                '%d de %b de %y',
                '%d de %m de %y',

                '%B %d, %Y',
                '%d %B %Y',
                '%d %b %Y',
                '%d %m %Y',
                '%d %B %y',
                '%d %b %y',
                '%d %m %y'):
        try:
            return datetime.datetime.strptime(dado, fmt)
        except ValueError:
            pass
    return np.nan


def tratar_dados_string(dado):
    dado_limpa = re.sub(r'[^a-zA-Z\s]', '', dado)

    # Remover espaços em branco à esquerda e à direita da string
    dado_limpa = dado_limpa.strip()
    
    return dado_limpa


def tratar_dados_numerico(coluna):
    
    n_dados = coluna.shape[0]
    if len(coluna.unique()) > 15:
        return (coluna - coluna.min()) / (coluna.max() - coluna.min())
    else:
        if coluna.dtypes in ['float32', 'float64', 'float128']:
            return coluna
        
        cat = {}
        for values in coluna.unique():
            cat[values] = len(cat)
        
        coluna = coluna.apply(lambda x: cat[x])
        return coluna

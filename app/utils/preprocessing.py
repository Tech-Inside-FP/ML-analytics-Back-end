# imports
import numpy as np
import datetime
import re


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

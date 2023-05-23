# imports
from utils.preprocessing import *
from utils.training import train

from fastapi.encoders import jsonable_encoder
from schemas.training import DatasetSchema
from fastapi import APIRouter

import pandas as pd
import base64


trainRouter = APIRouter()


# --- METHODS ---
async def read_request(data) -> dict:
    # decoding base64
    df_base64 = base64.b64decode(s=data['content'])
    # deconding utf-8
    df_bytes = df_base64.decode(encoding='utf-8')
    # converting to DataFrame
    df = pd.read_json(path_or_buf=df_bytes)
    
    # selecting columns
    df = df[data['features'] + [data['target']]]
    df_copy = df.copy(deep=True)
    
    # dropping nan rows
    df_copy.dropna(axis=0, inplace=True)
    
    # preprocessing methods
    preprocess_method = {
        'int32': tratar_dados_numerico,
        'int64': tratar_dados_numerico,
        'int128': tratar_dados_numerico,
        'float32': tratar_dados_numerico,
        'float64': tratar_dados_numerico,
        'float128': tratar_dados_numerico,
        'O': tratar_dados_string,
        'datetime64': tratar_coluna_data
    }
    
    # collecting columns
    columns = df_copy.columns
    column_dtype = df_copy.dtypes
    remove_col = []
    for col, dt in zip(columns, column_dtype):
        dt = str(dt)
        if dt in ['int32', 'int64', 'float32', 'float64']:
            df_copy[col] = preprocess_method[dt](df_copy[col])
        elif dt in ['O', 'datetime64']:
            df_copy[col] = df_copy[col].map(preprocess_method[dt])
        else:
            remove_col.append(col)
    df_copy = df_copy.drop(labels=remove_col, axis=1)
    train_df, test_df = split_dataset(data=df_copy)

    return train(model_type=data['model'], train=train_df, test=test_df)


# --- ROUTES ---
@trainRouter.get(path='/')
async def root() -> dict:
    return {'training-page': 'Running'}

@trainRouter.post(path='/')
async def deploy_dataset(request: DatasetSchema = (...)) -> dict:
    # reading request
    request = jsonable_encoder(obj=request)
    content = await read_request(data=request)

    return {'response': content}

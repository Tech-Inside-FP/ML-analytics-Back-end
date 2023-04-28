# imports
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
    
    return {
        'filename': data['filename'],
        'dataframe': df,
        'features': data['features'],
        'target': data['target']
    }

# --- ROUTES ---
@trainRouter.get(path='/')
async def root() -> dict:
    return {'training-page': 'Running'}

@trainRouter.post(path='/')
async def deploy_dataset(request: DatasetSchema = (...)) -> dict:
    # reading request
    request = jsonable_encoder(obj=request)
    content = await read_request(data=request)
    
    return {'response': list(content['dataframe'].columns)}
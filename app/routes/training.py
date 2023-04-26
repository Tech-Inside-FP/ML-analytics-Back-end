# imports
from schemas.training import DatasetSchema
from fastapi import APIRouter

import base64


trainRouter = APIRouter()


@trainRouter.get(path='/')
async def root() -> dict:
    return {'training-page': 'Running'}

@trainRouter.post(path='/')
async def deploy_dataset(file: DatasetSchema = (...)) -> dict:
    # extracting content
    content = file.content
    # decoding content
    decode = base64.b64decode(s=content)
    
    return {'response': decode.decode(encoding='utf-8')}
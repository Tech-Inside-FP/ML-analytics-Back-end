# imports
from schemas.training import DatasetSchema
from fastapi import APIRouter


trainRouter = APIRouter()


@trainRouter.get(path='/')
async def root():
    return {'training-page': 'Running!'}

@trainRouter.post(path='/')
async def deploy_dataset(file: DatasetSchema = (...)):
    pass
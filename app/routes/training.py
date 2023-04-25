# imports
from fastapi import APIRouter


trainRouter = APIRouter()


@trainRouter.post(path='/')
async def deploy_dataset(file: dict):
    pass
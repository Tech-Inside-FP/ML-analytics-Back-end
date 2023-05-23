# imports
from routes.training import trainRouter
from fastapi import FastAPI

import uvicorn


app = FastAPI()
app.include_router(router=trainRouter, prefix='/training')


@app.get(path='/')
def root():
    return {'ML-AnAIytics': 'Running'}

# imports
from fastapi import FastAPI

import uvicorn


app = FastAPI()


@app.get(path='/')
def root():
    return {'ML-AnAIytics': 'Running'}


if __name__=='__main__':
    uvicorn.run(app='main:app', host='0.0.0.0', port=8000, reload=True)
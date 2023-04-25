# imports
from pydantic import BaseModel, validator
from pydantic.types import Base64
from typing import Optional


class DatasetSchema(BaseModel):

    'filename': Optional[str]
    'content': Base64

    @validator('content')
    def val_content(cls, value):
        return value.decode()

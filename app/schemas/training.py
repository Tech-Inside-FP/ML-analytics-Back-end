# imports
from pydantic import BaseModel
from typing import Optional


class DatasetSchema(BaseModel):

    filename: Optional[str]
    content: bytes

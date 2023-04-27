# imports
from pydantic import BaseModel, Field
from typing import Optional, List

import base64


# --- MAIN SCHEMA ---
class DatasetSchema(BaseModel):

    filename: Optional[str]
    content: bytes = Field(...)
    features: List[str] = Field(...)
    target: str = Field(...)
    
    class Config:
        schema_extra = {
            'example': {
                'filename': 'database.csv',
                'content': base64.b64encode('dataframe.csv'.encode(encoding='utf-8')),
                'features': [
                    'column1',
                    'column3',
                    'column4',
                    'column7'
                ],
                'target': 'column8'
            }
        }

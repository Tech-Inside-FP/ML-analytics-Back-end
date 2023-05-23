# imports
from pydantic import BaseModel, Field
from typing import Optional, List

import pandas as pd
import base64


# --- MAIN SCHEMA ---
class DatasetSchema(BaseModel):

    filename: Optional[str]
    model: str = Field(...)
    content: bytes = Field(...)
    features: List[str] = Field(...)
    target: str = Field(...)
    
    class Config:
        schema_extra = {
            'example': {
                'filename': 'database.csv',
                'model': 'regression',
                'content': base64.b64encode(s=pd.read_csv(filepath_or_buffer='test_files/test_dataset_1.csv').to_json().encode(encoding='utf-8')),
                'features': [
                    'City',
                    'Rating',
                    'Votes'
                ],
                'target': 'Cost'
            }
        }

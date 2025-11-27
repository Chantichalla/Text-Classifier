from typing import Dict , Any
from pydantic import BaseModel

class UserInput(BaseModel):
    text : str
    class Config: 
        schema_extra = {
            # 
            "example_payload": {
                "text": "The new iphone is buggy"
            }
        }
class ModelResponse(BaseModel):
    label:str
    conf:str
class PredictionResponse(BaseModel):
    input : str
    models: Dict[str, ModelResponse]


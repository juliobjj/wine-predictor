from pydantic import BaseModel, conlist

class WineInput(BaseModel):
    features: conlist(float, min_length=11, max_length=11)
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field

class WineSampleInput(BaseModel):
    features: Annotated[list[float], Field(min_items=11, max_items=11)]

class WineSampleOutput(BaseModel):
    id: int
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float
    predicted_quality: int

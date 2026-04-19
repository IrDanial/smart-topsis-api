from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import numpy as np
from typing import List, Literal

# Init
app = FastAPI(
    title="SMART-TOPSIS API",
    description="Backend microservice for ranking alternatives.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"], # TODO Update this to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Validation
class TopsisRequest(BaseModel):
    alternatives: List[str]
    matrix: List[List[float]]
    criteria_points: List[float]
    criteria_types: List[Literal["cost", "benefit"]]

    @field_validator('matrix')
    def check_matrix(cls, matrix):
        if not matrix:
            raise ValueError("Matrix cannot be empty")
        cols = len(matrix[0])
        if any(len(row) != cols for row in matrix):
            raise ValueError("All rows in the matrix must have the same number of columns")
        return matrix

class RankedAlternative(BaseModel):
    rank: int
    alternative: str
    closeness_score: float

# Algorithm EndpointSS
@app.post("/api/rank", response_model=List[RankedAlternative])
def calculate_smart_topsis(payload: TopsisRequest):
    num_criteria = len(payload.criteria_points)
    
    # basic cross-validation
    if len(payload.matrix[0]) != num_criteria or len(payload.criteria_types) != num_criteria:
        raise HTTPException(status_code=400, detail="Matrix columns, points, and types must match in length.")
    if len(payload.alternatives) != len(payload.matrix):
        raise HTTPException(status_code=400, detail="Number of alternatives must match matrix rows.")

    points_array = np.array(payload.criteria_points)
    if np.sum(points_array) == 0:
        raise HTTPException(status_code=400, detail="Criteria points cannot all be zero.")
    weights = points_array / np.sum(points_array)

    matrix = np.array(payload.matrix)
    
    denominators = np.sqrt(np.sum(matrix**2, axis=0))
    denominators[denominators == 0] = 1
    normalized_matrix = matrix / denominators
    
    weighted_matrix = normalized_matrix * weights
    
    ideal = np.zeros(num_criteria)
    anti_ideal = np.zeros(num_criteria)
    
    for j in range(num_criteria):
        if payload.criteria_types[j] == 'benefit':
            ideal[j] = np.max(weighted_matrix[:, j])
            anti_ideal[j] = np.min(weighted_matrix[:, j])
        else:
            ideal[j] = np.min(weighted_matrix[:, j])
            anti_ideal[j] = np.max(weighted_matrix[:, j])
            
    dist_ideal = np.sqrt(np.sum((weighted_matrix - ideal)**2, axis=1))
    dist_anti = np.sqrt(np.sum((weighted_matrix - anti_ideal)**2, axis=1))
    
    denominator = dist_ideal + dist_anti
    closeness = np.divide(dist_anti, denominator, out=np.zeros_like(dist_anti), where=denominator!=0)

    results = [
        {"alternative": payload.alternatives[i], "closeness_score": float(closeness[i])}
        for i in range(len(payload.alternatives))
    ]
    
    results.sort(key=lambda x: x["closeness_score"], reverse=True)
    for i, res in enumerate(results):
        res["rank"] = i + 1

    return results
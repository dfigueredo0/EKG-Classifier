from fastapi import FastAPI
from pydantic import BaseModel
from ekgclf.inference import InferenceEngine, InferenceResult

app = FastAPI()

engine = InferenceEngine()

class EKGRequest(BaseModel):
    signal: list[float]  # or whatever your input is
class EKGResponse(BaseModel):
    class_label: str
    probabilities: dict[str, float]

@app.post("/predict", response_model=EKGResponse)
def predict_endpoint(req: EKGRequest):
    result = engine.predict(req.signal)
    return EKGResponse(**result)
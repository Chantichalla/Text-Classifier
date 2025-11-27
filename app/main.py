
from fastapi import FastAPI, HTTPException


from .Schemas import UserInput, PredictionResponse


from .services import load_models, predict_lstm, predict_bert

app = FastAPI(title="Universal AI Classifier", version="Functional")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
def home():
    return {"status": "online", "message": "API is running."}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(payload: UserInput):
    lstm_res = predict_lstm(payload.text)
    bert_res = predict_bert(payload.text)
    
    return {
        "input": payload.text,
        "models": {
            "LSTM": lstm_res,
            "BERT": bert_res
        }
    }


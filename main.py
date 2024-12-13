from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os
import time  # Не забудьте импортировать модуль time

app = FastAPI()

# Хранилище моделей
models = {}

# Схемы для API
class ModelConfig(BaseModel):
    id: str
    ml_model_type: str
    hyperparameters: dict

class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: ModelConfig

class LoadRequest(BaseModel):
    id: str

class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]

class RemoveResponse(BaseModel):
    message: str

@app.post("/fit", response_model=dict)
async def fit_model(request: FitRequest):
    config = request.config
    if config.id in models:
        raise HTTPException(status_code=400, detail="Model ID already exists")
    
    # Выбор типа модели
    if config.ml_model_type == "linear":
        model = LinearRegression(**config.hyperparameters)
    elif config.ml_model_type == "logistic":
        model = LogisticRegression(**config.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Model type not supported")

    # Обучение модели
    model.fit(request.X, request.y)

    # Задержка времени для выполнения требования задания
    time.sleep(60)
    
    models[config.id] = model

    # Сохранение модели на диск
    joblib.dump(model, f"{config.id}.model")
    return {"message": f"Model '{config.id}' trained and saved"}

@app.post("/load", response_model=dict)
async def load_model(config: LoadRequest):
    if config.id not in models:
        try:
            model = joblib.load(f"{config.id}.model")
            models[config.id] = model
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": f"Model '{config.id}' loaded"}

@app.post("/predict", response_model=dict)
async def predict(request: PredictRequest):
    if request.id not in models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    model = models[request.id]
    predictions = model.predict(request.X)
    return {"id": request.id, "predictions": predictions.tolist()}

@app.get("/list_models", response_model=List[str])
async def list_models():
    return list(models.keys())

@app.delete("/remove_all", response_model=RemoveResponse)
async def remove_all_models():
    models.clear()
    for file in os.listdir('.'):
        if file.endswith('.model'):
            os.remove(file)
    return {"message": "All models removed"}
import asyncio
import httpx
from time import time

async def fit_model(client, data):
    response = await client.post("http://localhost:8000/fit", json=data)
    return response.json()

async def run():
    async with httpx.AsyncClient() as client:
        model_data_1 = {
            "X": [[1, 2], [3, 4]],
            "y": [5, 6],
            "config": {
                "id": "linear_1",
                "ml_model_type": "linear",
                "hyperparameters": {"fit_intercept": True}
            }
        }

        model_data_2 = {
            "X": [[1, 2], [3, 4]],
            "y": [5, 6],
            "config": {
                "id": "linear_2",
                "ml_model_type": "linear",
                "hyperparameters": {"fit_intercept": True}
            }
        }

        # Обучение моделей
        start_time = time()
        responses = await asyncio.gather(
            fit_model(client, model_data_1),
            fit_model(client, model_data_2)
        )
        print("Training responses:", responses)
        print(f"Training duration: {time() - start_time} seconds")

        # Загрузка модели
        response = await client.post("http://localhost:8000/load", json={"id": "linear_1"})
        print("Load response:", response.json())

        # Прогнозирование
        response = await client.post(
            "http://localhost:8000/predict",
            json={"id": "linear_1", "X": [[5, 6], [7, 8]]}
        )
        print("Predict response:", response.json())

        # Получение списка моделей
        response = await client.get("http://localhost:8000/list_models")
        print("List models response:", response.json())

        # Удаление всех моделей
        response = await client.delete("http://localhost:8000/remove_all")
        print("Remove all models response:", response.json())

asyncio.run(run())
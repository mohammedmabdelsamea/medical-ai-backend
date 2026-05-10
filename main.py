from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 THIS IS THE KEY CHANGE (SPACE endpoint, not /models/)
HF_SPACE_URL = "https://YOUR-SPACE-NAME.hf.space/run/predict"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        # Spaces usually expect multipart file upload
        files = {"data": image}

        response = requests.post(
            HF_SPACE_URL,
            files=files,
            timeout=60
        )

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        if response.status_code != 200:
            return {
                "status": "error",
                "message": "Space request failed",
                "status_code": response.status_code,
                "raw": response.text
            }

        data = response.json()

        return {
            "status": "success",
            "result": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

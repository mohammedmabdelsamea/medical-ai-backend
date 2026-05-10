from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
HF_TOKEN = os.getenv("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok"}


def query_hf(image_bytes):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # IMPORTANT: HF sometimes needs retries (model loading)
    for attempt in range(3):
        response = requests.post(
            HF_API,
            headers=headers,
            data=image_bytes
        )

        # If model is loading, HF returns 503
        if response.status_code == 503:
            print("Model loading... retrying")
            time.sleep(2)
            continue

        return response

    return response


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = query_hf(image)

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        # If HF fails completely
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "HF API failed",
                "status_code": response.status_code,
                "raw": response.text
            }

        # Try JSON parse safely
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "Invalid JSON from HF",
                "raw": response.text
            }

        # HF error response
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
                "message": data["error"]
            }

        # Normal prediction
        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return {
                "status": "success",
                "prediction": top.get("label", "unknown"),
                "confidence": round(float(top.get("score", 0)), 4)
            }

        return {
            "status": "error",
            "message": "Unexpected HF response",
            "raw": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

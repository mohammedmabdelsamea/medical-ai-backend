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

HF_API = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"
HF_TOKEN = os.environ.get("HF_TOKEN")

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image
        )

        data = response.json()

        # FORCE SAFE OUTPUT FORMAT FOR LOVABLE
        if isinstance(data, list) and len(data) > 0:
            return {
                "prediction": str(data[0].get("label", "unknown")),
                "confidence": float(data[0].get("score", 0.0)),
                "raw": data
            }

        if isinstance(data, dict) and "error" in data:
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": data["error"]
            }

        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "error": "Unexpected response format",
            "raw": data
        }

    except Exception as e:
        return {
            "prediction": "error",
            "confidence": 0.0,
            "error": str(e)
        }

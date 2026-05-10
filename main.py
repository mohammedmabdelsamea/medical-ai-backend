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
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image,
            timeout=30
        )

        # Always try JSON safely
        try:
            data = response.json()
        except:
            return {
                "prediction": "error",
                "confidence": 0.0
            }

        # Valid Hugging Face output
        if isinstance(data, list) and len(data) > 0:
            return {
                "prediction": data[0].get("label", "unknown"),
                "confidence": float(data[0].get("score", 0.0))
            }

        # Hugging Face error case
        return {
            "prediction": "error",
            "confidence": 0.0
        }

    except:
        # NEVER break Lovable
        return {
            "prediction": "error",
            "confidence": 0.0
        }

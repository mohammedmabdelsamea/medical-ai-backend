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
            data=image
        )

        data = response.json()

        # 🧠 STRICT NORMALIZATION (IMPORTANT PART)

        if isinstance(data, list) and len(data) > 0:
            return {
                "prediction": str(data[0].get("label", "unknown")),
                "confidence": float(data[0].get("score", 0.0))
            }

        if isinstance(data, dict) and "error" in data:
            # STILL return valid structure (THIS FIXES LOVABLE)
            return {
                "prediction": "error",
                "confidence": 0.0
            }

        # fallback (still valid format!)
        return {
            "prediction": "unknown",
            "confidence": 0.0
        }

    except:
        # NEVER break frontend
        return {
            "prediction": "error",
            "confidence": 0.0
        }

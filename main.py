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

        # Try parse response safely
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "Invalid response from model"
            }

        # 🚨 CASE 1: model loading
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "loading",
                "message": data["error"]
            }

        # 🚨 CASE 2: valid prediction
        if isinstance(data, list) and len(data) > 0:
            top = data[0]

            return {
                "status": "success",
                "prediction": top.get("label"),
                "confidence": round(float(top.get("score", 0.0)), 3)
            }

        # 🚨 fallback (NO "unknown")
        return {
            "status": "error",
            "message": "Unexpected model output"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face model endpoint
HF_API = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"

# Token from Render environment
HF_TOKEN = os.environ.get("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = await file.read()

        # Call Hugging Face API
        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image,
            timeout=30
        )

        # Try to parse response safely
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "HF did not return JSON",
                "raw_response": response.text,
                "status_code": response.status_code
            }

        # 🔍 DEBUG: If HF returns error
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
                "message": data["error"],
                "hf_response": data
            }

        # ✅ SUCCESS CASE
        if isinstance(data, list) and len(data) > 0:
            top = data[0]

            return {
                "status": "success",
                "prediction": top.get("label", "unknown"),
                "confidence": float(top.get("score", 0.0)),
                "all_predictions": data
            }

        # ❌ Unexpected format
        return {
            "status": "error",
            "message": "Unexpected Hugging Face response format",
            "hf_response": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

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

# Hugging Face endpoint (correct inference API)
HF_API = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"

# Token from Render environment
HF_TOKEN = os.getenv("HF_TOKEN")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Backend is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = await file.read()

        # DEBUG INFO (IMPORTANT)
        print("HF_TOKEN EXISTS:", HF_TOKEN is not None)
        print("HF_API:", HF_API)

        # Call Hugging Face
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream"
            },
            data=image,
            timeout=30
        )

        print("HF STATUS CODE:", response.status_code)
        print("HF RAW RESPONSE:", response.text)

        # HTTP error from Hugging Face
        if response.status_code != 200:
            return {
                "status": "error",
                "stage": "hf_http_error",
                "status_code": response.status_code,
                "response": response.text
            }

        # Parse JSON safely
        try:
            data = response.json()
        except Exception:
            return {
                "status": "error",
                "stage": "invalid_json",
                "response": response.text
            }

        # Hugging Face model error (loading, etc.)
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
                "stage": "hf_model_error",
                "message": data["error"]
            }

        # Valid prediction
        if isinstance(data, list) and len(data) > 0:
            top = data[0]

            return {
                "status": "success",
                "prediction": top.get("label", "unknown"),
                "confidence": round(float(top.get("score", 0.0)), 4)
            }

        # Unexpected format
        return {
            "status": "error",
            "stage": "unexpected_format",
            "response": data
        }

    except Exception as e:
        return {
            "status": "error",
            "stage": "exception",
            "message": str(e)
        }

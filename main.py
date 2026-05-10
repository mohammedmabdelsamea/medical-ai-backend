from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Allow frontend (Lovable) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face model endpoint
HF_API = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"

# Secure token from Render environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Health check endpoint (useful for Render + debugging)
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Medical AI backend is live"
    }

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image = await file.read()

        # Call Hugging Face inference API
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            data=image
        )

        # Try parsing response safely
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "Hugging Face did not return JSON",
                "raw_response": response.text,
                "status_code": response.status_code
            }

        # Handle Hugging Face error response
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
                "message": data["error"],
                "status_code": response.status_code
            }

        # Handle valid prediction response
        if isinstance(data, list) and len(data) > 0:
            top = data[0]

            return {
                "status": "success",
                "prediction": top.get("label", "unknown"),
                "confidence": float(top.get("score", 0.0)),
                "all_predictions": data
            }

        # Fallback for unexpected format
        return {
            "status": "error",
            "message": "Unexpected response format from Hugging Face",
            "raw_response": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

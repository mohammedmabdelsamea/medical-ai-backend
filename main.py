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

# Correct Hugging Face Inference API endpoint
HF_API = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"

HF_TOKEN = os.environ.get("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = await file.read()

        # Call Hugging Face properly
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream"
            },
            data=image,
            timeout=30
        )

        # If request fails at HTTP level
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "Hugging Face API error",
                "status_code": response.status_code,
                "raw": response.text
            }

        # Try JSON parsing
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "Invalid JSON from Hugging Face",
                "raw": response.text
            }

        # Handle HF model loading or errors
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
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

        # Fallback (no fake "unknown")
        return {
            "status": "error",
            "message": "Unexpected model response format",
            "raw": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

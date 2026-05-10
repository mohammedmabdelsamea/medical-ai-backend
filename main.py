from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Allow Lovable frontend to access backend
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

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image = await file.read()

        # Call Hugging Face API
        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image
        )

        # Try to parse JSON safely
        try:
            result = response.json()
        except:
            return {
                "error": "Invalid JSON from Hugging Face",
                "raw_response": response.text
            }

        # Case 1: normal prediction (list of labels)
        if isinstance(result, list):
            return {
                "prediction": result[0]["label"],
                "confidence": result[0]["score"],
                "all_predictions": result
            }

        # Case 2: model loading / error response
        return {
            "error": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }

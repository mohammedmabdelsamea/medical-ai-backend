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

# ✅ API-ready Hugging Face model (IMPORTANT FIX)
HF_API = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

# Token from Render environment variables
HF_TOKEN = os.getenv("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image = await file.read()

        # Call Hugging Face Inference API
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            data=image,
            timeout=30
        )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        # Handle HTTP errors
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "Hugging Face API error",
                "status_code": response.status_code,
                "response": response.text
            }

        # Parse JSON
        data = response.json()

        # Handle model loading or HF errors
        if isinstance(data, dict) and "error" in data:
            return {
                "status": "error",
                "message": data["error"]
            }

        # Success response
        if isinstance(data, list) and len(data) > 0:
            top = data[0]

            return {
                "status": "success",
                "prediction": top.get("label", "unknown"),
                "confidence": round(float(top.get("score", 0.0)), 4)
            }

        return {
            "status": "error",
            "message": "Unexpected response format",
            "raw": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

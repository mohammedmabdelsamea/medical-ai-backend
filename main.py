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

# Hugging Face API endpoint
HF_API = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

# Token from Render environment
HF_TOKEN = os.getenv("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image = await file.read()

        # Debug logs (IMPORTANT for Render)
        print("HF_TOKEN EXISTS:", HF_TOKEN is not None)
        print("HF_API:", HF_API)

        # ✅ FIX: Hugging Face expects "files", NOT raw data
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            files={"file": image},
            timeout=30
        )

        print("HF STATUS CODE:", response.status_code)
        print("HF RESPONSE TEXT:", response.text)

        # If API fails
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "HF API error",
                "status_code": response.status_code,
                "response": response.text
            }

        # Try parsing JSON safely
        try:
            data = response.json()
        except:
            return {
                "status": "error",
                "message": "Invalid JSON from HF",
                "raw": response.text
            }

        # HF model error (loading, blocked, etc.)
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

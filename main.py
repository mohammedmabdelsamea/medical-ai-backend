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

# ✅ WORKING INFERENCE MODEL (IMPORTANT FIX)
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.get("/")
def root():
    return {"status": "ok", "message": "backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # read image
        image_bytes = await file.read()

        # send to Hugging Face Inference API
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes
        )

        print("HF STATUS:", response.status_code)
        print("HF RAW:", response.text[:500])

        # handle API failure
        if response.status_code != 200:
            return {
                "status": "error",
                "step": "hf_failed",
                "code": response.status_code,
                "raw": response.text[:300]
            }

        data = response.json()

        # HF returns list of predictions
        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return {
                "status": "success",
                "prediction": top.get("label"),
                "confidence": top.get("score")
            }

        return {
            "status": "error",
            "step": "unexpected_response",
            "raw": data
        }

    except Exception as e:
        return {
            "status": "error",
            "step": "backend_exception",
            "message": str(e)
        }

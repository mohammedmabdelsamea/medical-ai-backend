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

HF_API_URL = "https://api-inference.huggingface.co/models/HumaP/vit_base_patch16_224_in21k_lung_and_colon_histopathology_pt"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image
        )

        print("HF STATUS:", response.status_code)
        print("HF RAW:", response.text[:500])

        if response.status_code != 200:
            return {
                "status": "error",
                "step": "hf_failed",
                "code": response.status_code,
                "raw": response.text[:300]
            }

        result = response.json()

        # HF returns list of predictions
        top = result[0]

        return {
            "status": "success",
            "prediction": top.get("label"),
            "confidence": top.get("score")
        }

    except Exception as e:
        return {
            "status": "error",
            "step": "exception",
            "message": str(e)
        }

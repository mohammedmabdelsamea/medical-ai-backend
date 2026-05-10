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
HF_TOKEN = os.getenv("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        # FIXED REQUEST (IMPORTANT)
        response = requests.post(
            HF_API,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            data=image
        )

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        if response.status_code != 200:
            return {
                "status": "error",
                "message": response.text
            }

        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return {
                "status": "success",
                "prediction": top.get("label"),
                "confidence": float(top.get("score", 0))
            }

        return {
            "status": "error",
            "message": "Unexpected response",
            "raw": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

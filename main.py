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
HF_TOKEN = os.environ.get("HF_TOKEN")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image
        )

        # 🔍 DO NOT hide HF response anymore
        try:
            data = response.json()
        except:
            return {
                "error": "HF did not return JSON",
                "raw": response.text,
                "status_code": response.status_code
            }

        # Return EVERYTHING clearly
        return {
            "hf_status_code": response.status_code,
            "hf_response": data
        }

    except Exception as e:
        return {
            "error": str(e)
        }

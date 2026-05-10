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
def home():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        # Call Hugging Face
        response = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image
        )

        # Always capture raw text first (prevents crashes)
        raw_text = response.text

        # Try JSON parse safely
        try:
            data = response.json()
        except:
            return {
                "ok": False,
                "error": "HF returned non-JSON response",
                "raw": raw_text,
                "status_code": response.status_code
            }

        # Handle Hugging Face error format
        if isinstance(data, dict) and "error" in data:
            return {
                "ok": False,
                "error": data["error"],
                "status_code": response.status_code
            }

        # Handle normal classification output
        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return {
                "ok": True,
                "prediction": top.get("label"),
                "confidence": top.get("score"),
                "all": data
            }

        # Unexpected structure
        return {
            "ok": False,
            "error": "Unexpected HF response format",
            "raw": data
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IMPORTANT: use Space ROOT endpoint (NOT /run/predict)
HF_SPACE_URL = "https://mohammedabdelsamea-medical-ai-test.hf.space/api/predict"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # convert image to base64 (more reliable for Spaces)
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "data": [f"data:image/jpeg;base64,{encoded}"]
        }

        response = requests.post(
            HF_SPACE_URL,
            json=payload,
            timeout=60
        )

        print("HF STATUS:", response.status_code)
        print("HF RAW:", response.text[:1000])

        # fail-safe
        if response.status_code != 200:
            return {
                "status": "error",
                "step": "hf_request_failed",
                "code": response.status_code,
                "raw": response.text[:500]
            }

        try:
            data = response.json()
        except Exception:
            return {
                "status": "error",
                "step": "json_parse_failed",
                "raw": response.text[:500]
            }

        # Gradio usually returns: data["data"]
        result = None

        if isinstance(data, dict) and "data" in data:
            result = data["data"]
        else:
            result = data

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "step": "backend_exception",
            "message": str(e)
        }

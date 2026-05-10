from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_SPACE_URL = "https://mohammedabdelsamea-medical-ai-test.hf.space/run/predict"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        response = requests.post(
            HF_SPACE_URL,
            files={"data": image},
            timeout=60
        )

        # If Space fails
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "Space request failed"
            }

        data = response.json()

        # 🧠 NORMALISE RESPONSE FOR LOVABLE
        # (this is the critical fix)

        if isinstance(data, dict):
            prediction = data.get("prediction", "unknown")
            confidence = data.get("confidence", 0.0)
        else:
            prediction = str(data)
            confidence = 0.0

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

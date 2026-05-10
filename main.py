from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

# Allow Lovable frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Hugging Face Space endpoint
HF_SPACE_URL = "https://mohammedabdelsamea-medical-ai-test.hf.space/run/predict"


@app.get("/")
def root():
    return {"status": "ok", "message": "backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from request
        image_bytes = await file.read()

        # Send to Hugging Face Space
        response = requests.post(
            HF_SPACE_URL,
            files={"data": image_bytes},
            timeout=60
        )

        # If Hugging Face fails
        if response.status_code != 200:
            return {
                "status": "error",
                "step": "hf_request_failed",
                "status_code": response.status_code,
                "response_text": response.text[:500]
            }

        # Try to parse JSON safely
        try:
            data = response.json()
        except Exception:
            return {
                "status": "error",
                "step": "json_parse_error",
                "response_text": response.text[:500]
            }

        # Normalize response so Lovable NEVER breaks
        prediction = data.get("prediction", "unknown") if isinstance(data, dict) else str(data)
        confidence = data.get("confidence", 0.0) if isinstance(data, dict) else 0.0

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "status": "error",
            "step": "backend_exception",
            "message": str(e)
        }

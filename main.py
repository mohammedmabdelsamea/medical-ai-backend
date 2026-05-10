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
    image = await file.read()

    response = requests.post(
        HF_SPACE_URL,
        files={"data": image}
    )

    return response.json()

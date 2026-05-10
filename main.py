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

# 🔥 YOUR SPACE URL (replace this)
HF_SPACE_URL = mohammedabdelsamea/medical-ai-test


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        # Spaces expect "files"
        response = requests.post(
            HF_SPACE_URL,
            files={"data": image}
        )

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        if response.status_code != 200:
            return {
                "status": "error",
                "message": "Space error",
                "raw": response.text
            }

        return {
            "status": "success",
            "result": response.json()
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

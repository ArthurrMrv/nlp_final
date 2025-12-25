import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Model Configuration
MODEL_ID = "ArthurMrv/deberta-v3-ft-financial-news-sentiment-analysis-finetuned"
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize pipeline (lazy loading might be better for startup speed, but we'll load on start for readiness)
print("Loading model...")
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=HF_TOKEN,
        return_all_scores=True
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # The pipeline returns a list of lists of dicts for single text input [[{'label': '...', 'score': ...}, ...]]
        # or a list of dicts depending on version/config. We assume standard return.
        predictions = classifier(request.text)
        
        # Normalize response if needed (e.g., sorting by score)
        # Assuming predictions is a list of dicts or list of list of dicts.
        # Let's return the raw predictions for flexibility in the frontend
        return {"sentiment": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (Frontend)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

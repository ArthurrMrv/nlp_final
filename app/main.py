import os
import yfinance as yf
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

# Initialize pipeline
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

class TickerRequest(BaseModel):
    symbol: str

@app.post("/predict")
async def predict(request: TextRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = classifier(request.text)
        return {"sentiment": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_ticker")
async def analyze_ticker(request: TickerRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch news
        ticker = yf.Ticker(request.symbol)
        news = ticker.news
        
        if not news:
            return {"data": []}
            
        # Prepare data
        items_to_analyze = []
        texts = []
        
        # Limit to 5-10 latest news to keep it fast
        for item in news[:5]:
            title = item.get('title', '')
            if title:
                items_to_analyze.append(item)
                texts.append(title)
        
        if not texts:
            return {"data": []}

        # Predict
        predictions = classifier(texts)
        
        # Combine
        results = []
        for item, sent in zip(items_to_analyze, predictions):
            # sent is a list of scores like [{'label': '..', 'score': ..}, ..]
            results.append({
                "title": item.get('title'),
                "link": item.get('link'),
                "publisher": item.get('publisher'),
                "published": item.get('providerPublishTime'),
                "sentiment": sent
            })
            
        return {"data": results}

    except Exception as e:
        print(f"Error processing ticker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (Frontend)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
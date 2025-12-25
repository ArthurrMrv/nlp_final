import streamlit as st
import requests
import os
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://z64b4h1haoq2ctat.eu-west-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv("HF_TOKEN")

# Headers for the API request
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            return {"error": f"403 Forbidden. Please check if your HF_TOKEN is valid and has permission to access this endpoint. Token loaded: {'Yes' if HF_TOKEN else 'No'}"}
        return {"error": str(e)}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def extract_news_content(item):
    """Extract title, link, and publisher from yfinance news item."""
    content = item.get('content', {})
    
    title = content.get('title', '')
    
    # Get link from canonicalUrl or clickThroughUrl
    link = None
    if content.get('canonicalUrl'):
        link = content['canonicalUrl'].get('url')
    elif content.get('clickThroughUrl'):
        link = content['clickThroughUrl'].get('url')
    
    # Get publisher from provider
    publisher = None
    if content.get('provider'):
        publisher = content['provider'].get('displayName')
    
    return title, link, publisher


def analyze_news(news_items, status_placeholder=None):
    results = []
    total = len(news_items)
    
    for i, item in enumerate(news_items):
        title, link, publisher = extract_news_content(item)
        if not title:
            continue
            
        if status_placeholder:
            status_placeholder.info(f"🔄 Analyzing news {i+1}/{total}: *{title[:60]}...*")
            
        sentiment_response = query({"inputs": title})
        
        # Check if response is an error
        if isinstance(sentiment_response, dict) and "error" in sentiment_response:
             results.append({
                "title": title,
                "link": link,
                "publisher": publisher,
                "sentiment": "Error",
                "score": 0,
                "details": sentiment_response['error']
            })
        else:
            try:
                if isinstance(sentiment_response, list) and isinstance(sentiment_response[0], list):
                    top_sentiment = max(sentiment_response[0], key=lambda x: x['score'])
                elif isinstance(sentiment_response, list) and isinstance(sentiment_response[0], dict):
                     top_sentiment = max(sentiment_response, key=lambda x: x['score'])
                elif isinstance(sentiment_response, dict):
                    top_sentiment = sentiment_response
                else:
                    top_sentiment = {"label": "Unknown", "score": 0}

                results.append({
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "sentiment": top_sentiment.get('label', 'Unknown'),
                    "score": top_sentiment.get('score', 0)
                })
            except Exception as e:
                results.append({
                "title": title,
                "link": link,
                "publisher": publisher,
                "sentiment": "Parse Error",
                "score": 0,
                "details": str(e)
            })
            
    if status_placeholder:
        status_placeholder.empty()
    return results

def main():
    st.set_page_config(page_title="Financial Sentiment Analysis", layout="wide")
    st.title("Financial Sentiment Analysis")

    # Sidebar
    st.sidebar.header("Configuration")
    if not HF_TOKEN:
        st.sidebar.error("HF_TOKEN is missing in .env file!")
    else:
        st.sidebar.success("HF_TOKEN loaded.")

    mode = st.sidebar.radio("Select Mode", ["Stock News", "Custom Text"])

    if mode == "Custom Text":
        st.header("Analyze Custom Text")
        user_input = st.text_area("Input Text", height=200, placeholder="Enter financial news or text here...")

        if st.button("Analyze"):
            if user_input.strip():
                status_log = st.empty()
                status_log.info("📡 Connecting to Hugging Face Inference Server...")
                with st.spinner('Processing...'):
                    output = query({"inputs": user_input})
                    status_log.empty()
                    
                    if isinstance(output, dict) and "error" in output:
                        st.error(f"Error: {output['error']}")
                    else:
                        st.success("Analysis Complete!")
                        st.json(output)
            else:
                st.warning("Please enter some text to analyze.")

    elif mode == "Stock News":
        st.header("Analyze Stock News")
        
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "BAC", "BTC-USD", "ETH-USD"]
        selected_ticker = st.selectbox("Select Ticker", tickers)
        
        # Initialize session state for results if it doesn't exist
        if 'news_results' not in st.session_state:
            st.session_state['news_results'] = None
        if 'current_ticker' not in st.session_state:
            st.session_state['current_ticker'] = None

        if st.button(f"Get News & Analyze for {selected_ticker}"):
            # Clear previous results
            st.session_state['news_results'] = None
            st.session_state['current_ticker'] = selected_ticker
            
            # Log Container
            log_container = st.empty()
            logs = []

            def update_logs(message):
                logs.append(message)
                # Join with newlines to look like a terminal
                log_container.code("\n".join(logs), language="bash")

            try:
                update_logs(f"🚀 Starting analysis for {selected_ticker}...")
                update_logs("⏳ Fetching latest news from Yahoo Finance...")
                
                ticker_data = yf.Ticker(selected_ticker)
                news = ticker_data.news
                
                if not news:
                    update_logs("⚠️ No news found for this ticker.")
                    st.warning("No news found.")
                else:
                    update_logs(f"✅ Found {len(news)} news items.")
                    # Limit to latest 5
                    news_to_analyze = news[:5]
                    update_logs(f"📉 Selecting latest {len(news_to_analyze)} items for analysis.")
                    
                    # Custom analysis loop with logging
                    analyzed_data = []
                    update_logs("🧠 Warming up inference model...")
                    
                    for i, item in enumerate(news_to_analyze):
                        title, link, publisher = extract_news_content(item)
                        if not title:
                            continue
                            
                        update_logs(f"🔍 Analyzing {i+1}/{len(news_to_analyze)}: {title[:50]}...")
                        
                        # Call API
                        sentiment_response = query({"inputs": title})
                        
                        # Process response
                        if isinstance(sentiment_response, dict) and "error" in sentiment_response:
                             analyzed_data.append({
                                "title": title,
                                "link": link,
                                "publisher": publisher,
                                "sentiment": "Error",
                                "score": 0,
                                "details": sentiment_response['error']
                            })
                        else:
                            try:
                                if isinstance(sentiment_response, list) and isinstance(sentiment_response[0], list):
                                    top_sentiment = max(sentiment_response[0], key=lambda x: x['score'])
                                elif isinstance(sentiment_response, list) and isinstance(sentiment_response[0], dict):
                                     top_sentiment = max(sentiment_response, key=lambda x: x['score'])
                                elif isinstance(sentiment_response, dict):
                                    top_sentiment = sentiment_response
                                else:
                                    top_sentiment = {"label": "Unknown", "score": 0}

                                analyzed_data.append({
                                    "title": title,
                                    "link": link,
                                    "publisher": publisher,
                                    "sentiment": top_sentiment.get('label', 'Unknown'),
                                    "score": top_sentiment.get('score', 0)
                                })
                            except Exception as e:
                                analyzed_data.append({
                                    "title": title,
                                    "link": link,
                                    "publisher": publisher,
                                    "sentiment": "Parse Error", 
                                    "score": 0, 
                                    "details": str(e)
                                })
                    
                    update_logs("✅ Analysis complete. Formatting results...")
                    st.session_state['news_results'] = analyzed_data
                    
            except Exception as e:
                update_logs(f"❌ Error: {str(e)}")
                st.error(f"Error fetching data: {e}")
            
            # Clear logs after a brief moment or immediately if preferred. 
            # User requested it disappears when job completed.
            # log_container.empty()

        # Display Results (Outside button logic so it persists)
        if st.session_state['news_results'] is not None and st.session_state['current_ticker'] == selected_ticker:
            st.subheader(f"Latest Analysis for {st.session_state['current_ticker']}")
            results = st.session_state['news_results']
            
            if not results:
                st.info("No results to display.")
            
            for res in results:
                # Color code based on sentiment
                label = res['sentiment'].lower()
                icon = "⚪"
                if "bullish" in label or "positive" in label:
                    icon = "🟢"
                elif "bearish" in label or "negative" in label:
                    icon = "🔴"
                elif "neutral" in label:
                    icon = "🟡"

                with st.expander(f"{icon} {res['sentiment']} ({res['score']:.2f}): {res['title']}"):
                    st.write(f"**Publisher:** {res.get('publisher', 'Unknown')}")
                    if res.get('link'):
                        st.write(f"**Link:** [Read more]({res['link']})")
                    if "details" in res:
                        st.error(f"Error details: {res['details']}")

if __name__ == "__main__":
    main()

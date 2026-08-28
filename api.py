"""
Router.py is transformed into a legitimate web API using api.py.

Without requiring Python or your model files to be installed, 
anyone (or any app, or your Streamlit demo) may send a URL or email to it over the internet or network and receive a phishing prediction back. 
This distinguishes it as a true "product" as opposed to merely a manually executed script.

HOW TO RUN THIS FILE:
    uvicorn api:app --reload

Then open this in your browser to see interactive API docs:
    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel

from router import predict

app = FastAPI(
    title="Phishing Detection API",
    description="Send a URL or email body and get back a phishing probability.",
)


# This defines what a valid request looks like: a JSON object with one
# field called "text". FastAPI uses this to validate incoming requests
# automatically and reject anything malformed.
class PredictRequest(BaseModel):
    text: str


@app.get("/")
def root():
    """A simple homepage so you know the API is alive if you visit it directly."""
    return {
        "message": "Phishing Detection API is running.",
        "usage": "POST a JSON body like {'text': 'your url or email here'} to /predict",
    }


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    """The main endpoint. Takes text in, returns the full prediction breakdown."""
    return predict(request.text)

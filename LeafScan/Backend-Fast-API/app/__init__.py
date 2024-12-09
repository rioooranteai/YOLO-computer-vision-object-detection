from fastapi import FastAPI, Request
from .routes import main
from prometheus_client import generate_latest, Histogram
import time

def create_app():
    app = FastAPI()

    app.include_router(main)

    return app

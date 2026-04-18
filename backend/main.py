from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.course_api import router as course_router

app = FastAPI()

app.include_router(course_router)

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/downloads", StaticFiles(directory=str(DOWNLOADS_DIR)), name="downloads")

@app.get("/")
def root():
    return {"msg": "Teaching Agent Backend Running"}

from fastapi import FastAPI
from api.rag_api import router as rag_router
from api.course_api import router as course_router

app = FastAPI()

app.include_router(rag_router)
app.include_router(course_router)

@app.get("/")
def root():
    return {"msg": "Teaching Agent Backend Running"}
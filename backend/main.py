from fastapi import FastAPI
from api.rag_api import router as rag_router
from api.course_api import router as course_router
from api.syllabus_api import router as syllabus_router
from api.document_api import router as document_router
from api.kb_api import router as kb_router

app = FastAPI()

app.include_router(rag_router)
app.include_router(course_router)
app.include_router(syllabus_router)
app.include_router(document_router)
app.include_router(kb_router)

@app.get("/")
def root():
    return {"msg": "Teaching Agent Backend Running"}

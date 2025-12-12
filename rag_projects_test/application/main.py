from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from minio import Minio
import os
import pandas as pd

from utils import LoggingLogicFunctions as llf
from utils import DocEduc
from utils import BacklogAction

from rag.rag_utils import MinioEngine as me
from rag.rag_utils import RagUtils as ru
from rag.main import RagChat

# --------------------------
#  ⚙️ CONFIG
# --------------------------
MINIO_HOST = os.getenv("MINIO_HOST", "localhost:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "admin123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false") == "true"

BUCKET = "documents-edu"

app = FastAPI()

# Templates / Static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------------------------
#  📦 INIT MINIO ENGINE
# --------------------------
engine = me(
    host=MINIO_HOST,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=MINIO_SECURE
).object_engine()

if not engine.bucket_exists(BUCKET):
    engine.make_bucket(BUCKET)

# --------------------------
#  🤖 INIT RAG
# --------------------------
rag_chat = RagChat(minio_engine=engine)


# --------------------------
#  404 CUSTOM
# --------------------------
@app.exception_handler(404)
async def not_found_page(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


# --------------------------
#  CHAT UI
# --------------------------
@app.get("/chat_ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/chat")
async def chatbot(prompt: str):
    response = rag_chat.chat(prompt=prompt)
    return {"message": response.get('message')}


# --------------------------
#  ADMIN HTML PAGES
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def dashboard(request: Request):
    docs_count = len([obj.object_name for obj in engine.list_objects(BUCKET)])

    if os.path.exists("backlog.txt"):
        number_of_requests = pd.read_csv("backlog.txt").shape[0]
    else:
        number_of_requests = 0

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard",
        "doc_actifs": docs_count,
        "number_of_requests": number_of_requests
    })


@app.get("/admin_integration", response_class=HTMLResponse)
async def integration(request:Request):
    return templates.TemplateResponse("integration.html", {"request":request})

# --------------------------
#  📥 REFRESH DOC
# --------------------------
@app.get("/refresh")
async def refresh(request:Request):
    ru.create_base_of_knowledge(minio_engine=engine, bucket_name=BUCKET, nb_chuncks=2500)
    return {"message":"Data refreshed !"}



# --------------------------
#  📥 UPLOAD DOC
# --------------------------
@app.post("/add_doc")
async def upload_document(
    course: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...)
):
    engine.put_object(
        bucket_name=BUCKET,
        object_name=file.filename,
        data=file.file,
        length=-1,
        part_size=10 * 1024 * 1024,
    )

    minio_path = f"{BUCKET}/{file.filename}"
    doc = DocEduc(course, description, path=minio_path)

    llf.acting_backlog(document=doc, action=BacklogAction.ADD.value)
    llf.acting_checkpoints()

    return {"status": "ok", "file": file.filename}


# --------------------------
#  🗑 DELETE DOC
# --------------------------
@app.delete("/delete_doc")
async def delete_document(filename: str = Form(...)):
    objects = [obj.object_name for obj in engine.list_objects(BUCKET)]

    if filename not in objects:
        return {"message": "file not exists"}

    engine.remove_object(BUCKET, filename)

    df_checkpoint = pd.read_csv("checkpoints.csv")
    row = df_checkpoint[df_checkpoint["path_bucket"] == f"{BUCKET}/{filename}"].to_dict("records")[0]

    doc = DocEduc(row["course"], row["description"], row["path_bucket"])

    llf.acting_backlog(document=doc, action=BacklogAction.DELETE.value)
    llf.acting_checkpoints()

    return {"status": "deleted", "file": filename}


# --------------------------
#  📄 LIST DOCS
# --------------------------
@app.get("/list_doc")
async def list_documents():
    if not os.path.exists("checkpoints.csv"):
        return {"documents": []}

    objects = engine.list_objects(BUCKET)

    df_obj = pd.DataFrame([{"path_bucket": f"{BUCKET}/{o.object_name}"} for o in objects])
    df_check = pd.read_csv("checkpoints.csv")

    df_merged = df_obj.merge(df_check, on="path_bucket", how="left")
    df_merged.drop(columns=["log"], inplace=True)

    return {"documents": df_merged.to_dict(orient="records")}


# --------------------------
#  RUN SERVER
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

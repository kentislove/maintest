import os
import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from utils import load_documents_from_folder
import gradio as gr


VECTOR_STORE_PATH = "./faiss_index"
DOCUMENTS_PATH = "./docs"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None

def build_vector_store():
    documents = load_documents_from_folder(DOCUMENTS_PATH)
    if not documents:
        return "資料夾內沒有文件，請先上傳文件"
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(VECTOR_STORE_PATH)
    global vectorstore
    vectorstore = db
    return "向量庫建立成功"

def load_vector_store():
    global vectorstore
    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)

def similarity_search(query):
    load_vector_store()
    if vectorstore is None:
        return "向量庫不存在，請先建立向量庫"
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "查無相似資料"
    results = "\n\n".join([doc.page_content for doc in docs])
    return results

def save_uploaded_files(files):
    allowed_exts = {".doc", ".docx", ".xls", ".xlsx", ".pdf", ".txt"}
    saved_files = []
    if not files:
        return "請選擇文件"
    if not isinstance(files, list):
        files = [files]
    for f in files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in allowed_exts:
            continue
        save_path = os.path.join(DOCUMENTS_PATH, os.path.basename(f.name))
        shutil.copy(f.name, save_path)
        saved_files.append(os.path.basename(f.name))
    if saved_files:
        return f"檔案已上傳，請點「建立向量庫」\n已上傳: {', '.join(saved_files)}"
    else:
        return "未上傳有效文件"

with gr.Blocks(title="純向量搜尋 AI BOX") as demo:
    query = gr.Textbox(label="請輸入查詢")
    submit = gr.Button("查詢")
    answer = gr.Textbox(label="搜尋結果", interactive=False, show_copy_button=True)
    upload = gr.File(label="上傳文件", file_types=[".doc", ".docx", ".xls", ".xlsx", ".pdf", ".txt"], file_count="multiple")
    upload_btn = gr.Button("上傳")
    build_btn = gr.Button("建立向量庫")
    status = gr.Markdown()

    submit.click(fn=similarity_search, inputs=query, outputs=answer)
    upload_btn.click(fn=save_uploaded_files, inputs=upload, outputs=status)
    build_btn.click(fn=build_vector_store, inputs=None, outputs=status)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/gradio")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/gradio")

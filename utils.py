import os
from typing import List
from langchain.schema import Document
from langchain.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(filepath, autodetect_encoding=True)
            elif file.endswith(".pdf"):
                loader = UnstructuredPDFLoader(filepath)
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(filepath)
            elif file.endswith(".xlsx") or file.endswith(".xls"):
                loader = UnstructuredExcelLoader(filepath)
            else:
                print(f"不支援的格式：{file}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"讀取失敗 {file}: {e}")
    return docs

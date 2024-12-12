from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdfs(file):
    loader = DirectoryLoader(file, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Document Loaded\n\n")
    return documents
    
# split text into chunks
def text_split(extract_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    chunks = text_splitter.split_documents(extract_data)
    print("Chunks Splited\n\n")

    return chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("Embeddings Downloaded\n\n")
    return embeddings
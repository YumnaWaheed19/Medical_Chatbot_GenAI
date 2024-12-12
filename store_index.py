
# Before launching your application - for the very first time you've to execute this store_index file 
# run only one time
from src.helper import load_pdfs, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY is not set!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdfs(file='D:\\GenAI\\Medical_Chatbot_GenAI\\Data\\')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
print(index_name," index name")
pc.create_index(
                name=index_name,
                dimension=384, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )


docsrearch = PineconeVectorStore.from_documents(
            index_name=index_name,
            embedding=embeddings, 
            documents=text_chunks
        )

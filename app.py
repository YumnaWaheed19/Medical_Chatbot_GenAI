from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
# Load Existing Index
docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name, 
            embedding=embeddings, 
        )
           
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
retrieved_docs = retriever.invoke("What is acne?")

# print("docss: ",retrieved_docs)
# print("Response : ", retrieved_docs["answer"])
llm = OllamaLLM(model="llama2",  
                temperature = 0.3
            )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No input provided!"}), 400

    print("User message:", msg)
    response = rag_chain.invoke({"input": msg})
    print("RAG Response:", response)

    # Adjust based on response structure
    return str(response.get("answer", "No answer found"))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
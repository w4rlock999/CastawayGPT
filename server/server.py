from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document
from typing import List
import json
from dotenv import load_dotenv
import os

# Load API_KEY from .env file 
load_dotenv()
ENV_OpenAI_api_key = os.getenv("API_KEY")

app = Flask(__name__)    

@app.route("/", methods=['GET'])
def Root():
    return jsonify({"message": "Welcome?"})

@app.route("/addEmbeddingToChroma", methods=['POST'])
def AddEmbeddingToChroma():
    try:    
        json_data = request.get_json()                          
        docs: List[Document] = []
        for curDocument in json_data:
            page_content = curDocument["page_content"]            
            metadata = curDocument["metadata"]
                        
            docs.append(Document(page_content=page_content, metadata=metadata['timestamp']))
        print(docs)

        # create the open-source embedding function -> change to openAI
        # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)

        # load it into Chroma
        db = Chroma.from_documents(collection_name = "transcript_db",documents = docs, embedding = embedding_function)

        # query it
        query = "Where is the speaker come from?"
        docs = db.similarity_search(query)

        # print results
        print(docs[0].page_content)
        print(docs[1].page_content)
        print(docs[2].page_content)

        return jsonify({"message": "Message Received Correctly"})
    
    except Exception as e:    
        return jsonify({"error": str(e)})

@app.route("/chatWithContext", methods=['POST'])
def ChatWithContext():

    try:
        json_data = request.get_json()
        print("context search: " + json_data)
        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
        db = Chroma(collection_name="transcript_db", embedding_function = embedding_function)
                
        docs = db.similarity_search(json_data)

        print(docs)
        # print results
        # print(docs[0].page_content)
        # print(docs[1].page_content)
        # print(docs[2].page_content)

        return jsonify({"response" : "success"})
    except Exception as e:
        return jsonify({"error": str(e)})


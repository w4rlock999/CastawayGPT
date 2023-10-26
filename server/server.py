from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document
from typing import List
import json

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

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        db = Chroma.from_documents(docs, embedding_function)

        # query it
        query = "What is the background of the speaker?"
        docs = db.similarity_search(query)

        # print results
        print(docs[0].page_content)

        return jsonify({"message": "Message Received Correctly"})
    
    except Exception as e:    
        return jsonify({"error": str(e)})


@app.route("/chatWithContext", methods=['POST'])
def ChatWithContext():
    return jsonify({"AIMessage" : "I, Robot"})

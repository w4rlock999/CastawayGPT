from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain import OpenAI
from typing import List
import json
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from langchain.chains.summarize import load_summarize_chain


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
            print(curDocument)
            page_content = curDocument["page_content"]            
            metadata = curDocument["metadata"]
                        
            docs.append(Document(page_content=page_content, metadata=metadata['timestamp']))
        # print(docs)
        llm = OpenAI(temperature=0, openai_api_key=ENV_OpenAI_api_key)
        num_tokens = llm.get_num_tokens(docs[0].page_content)

        print(f"received {num_tokens} tokens")

        # create the open-source embedding function -> change to openAI
        # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)

        # load it into Chroma
        db = Chroma.from_documents(collection_name = "transcript_db", documents = docs, embedding = embedding_function)        
        vectors = db.get()
        print(vectors)

        return jsonify({"message": "Transcript Received Correctly"})
    
    except Exception as e:    
        return jsonify({"error": str(e)})

@app.route("/summarizeVectorData", methods=['POST'])
def SummarizeVectorData():
    try:
        json_data = request.get_json()    

        videoInfo = json_data['videoInfo']
        transcriptDocuments = json_data['transcriptDocuments']

        docs: List[Document] = []

        for curDocument in transcriptDocuments:            
            page_content = curDocument["page_content"]            
            metadata = curDocument["metadata"]                        
            docs.append(Document(page_content=page_content, metadata=metadata['timestamp']))

        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
        vectors = embedding_function.embed_documents([x.page_content for x in docs])

        # print(vectors)
        num_clusters = 11
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

        kmeans.labels_
        print("KMeans calculated")
        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt

        # # Taking out the warnings
        # import warnings
        # from warnings import simplefilter

        # # Filter out FutureWarnings
        # simplefilter(action='ignore', category=FutureWarning)

        # # Perform t-SNE and reduce to 2 dimensions
        # tsne = TSNE(n_components=2, random_state=42)
        # reduced_data_tsne = tsne.fit_transform(vectors)

        # # Plot the reduced data
        # plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.title('Book Embeddings Clustered')
        # plt.show()
        
        closest_indices = []

        for i in range(num_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        selected_indices = sorted(closest_indices)

        selected_docs = [docs[doc] for doc in selected_indices]
        # print(selected_docs)
        # print(len(selected_docs))
        llm = OpenAI(temperature=0, openai_api_key=ENV_OpenAI_api_key)
        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',
                                    )
        output = summary_chain.run(docs)
        print(output)
        
        response_data = {
            "title": videoInfo["videoTitle"],
            "content": output,
            "videos": [
                {"title": "Example Video 1", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"},
                {"title": "Example Video 2", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=700"}
            ]
        }
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/chatWithContext", methods=['POST'])
def ChatWithContext():
    try:
        json_data = request.get_json()
        print(json_data['message'])
        # print("request message: " + json_data.message)

        print("")
        print("context search: " + json_data['message'])
        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
        db = Chroma(collection_name="transcript_db", embedding_function = embedding_function)      

        # search vector store 
        docs = db.similarity_search(json_data['message'])        
        # print results        
        print(docs[0].page_content)
        print(docs[1].page_content)
        print(docs[2].page_content)

        response_data = {
            "title": "Example Response",
            "content": "This is a sample response with video timestamps.",
            "videos": [
                {"title": "Video 1", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"},
                {"title": "Video 2", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=700"}
            ]
        }
        return jsonify(response_data)
        return jsonify({"response" : "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)})


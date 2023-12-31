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
from typing import Optional
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load API_KEY from .env file 
load_dotenv()
ENV_OpenAI_api_key = os.getenv("API_KEY")

app = Flask(__name__)    

def ConvertTimestampMetadataIntoSeconds(metadata):
    outputTimestamp = 0
    outputTimestamp += metadata['hours'] * 3600 
    outputTimestamp += metadata['minutes'] * 60 
    outputTimestamp += metadata['seconds']
    return outputTimestamp


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

        llm = OpenAI(temperature=0, openai_api_key=ENV_OpenAI_api_key)        

        # create the open-source embedding function
        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)

        # load it into Chroma
        db = Chroma.from_documents(collection_name = "transcript_db", documents = docs, embedding = embedding_function)        

        return jsonify({"message": "Transcript embedded into ChromaDB successfully"})
    
    except Exception as e:    
        return jsonify({"error": str(e)})

@app.route("/summarizeVectorData", methods=['POST'])
def SummarizeVectorData():
    try:
        json_data = request.get_json()    

        videoInfo = json_data['videoInfo']

        print("video ID: ")
        print(videoInfo["videoID"])
        videoID = videoInfo["videoID"]

        transcriptDocuments = json_data['transcriptDocuments']

        docs: List[Document] = []

        for curDocument in transcriptDocuments:            
            page_content = curDocument["page_content"]            
            metadata = curDocument["metadata"]                        
            docs.append(Document(page_content=page_content, metadata=metadata['timestamp']))

        embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
        vectors = embedding_function.embed_documents([x.page_content for x in docs])
        
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)        
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
        
        # find each cluster's centroid 
        closest_indices = []

        for i in range(num_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        selected_indices = sorted(closest_indices)
        selected_docs = [docs[doc] for doc in selected_indices]
        
        print(f"found {len(selected_docs)} centroids documents: ")
        print(selected_docs)        

        # create summary from centroids documents
        llm = OpenAI(temperature = 0, openai_api_key = ENV_OpenAI_api_key)
        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')
        summaryOutput = summary_chain.run(selected_docs)
        
        print("summary: ")
        print(summaryOutput)
        
        json_schema = {
            "title": "Topics",
            "description": "Generate various unique topics from a passage.",
            "type": "object",
            "properties": {
                "topics": 
                {   
                    "description" : "array of topics generated from the passage",
                    "type" : "array", 
                    "items"  : {
                        "description" : "one unique topic generated from the passage",
                        "type" : "string"
                    }
                },
            },
            "required": ["name", "age"],
        }
        promptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class algorithm for extracting topics from a passage in structured formats.",
                ),
                (
                    "human", "Use the given format for 5 and only 5 topics, with each topic in 3 to 5 words, from the following input: {input}",
                ),
                (   
                    "human", "Tip: Make sure to answer in the correct format and only give 5 topics, no more, no less"
                ),
            ]
        )

        llmChat = ChatOpenAI(model = "gpt-3.5-turbo-0613", temperature = 0, openai_api_key = ENV_OpenAI_api_key)
        runnable = create_structured_output_chain(json_schema, llmChat, promptTemplate)
        topics = runnable.invoke({"input": summaryOutput})
        print(topics)
        
        db = Chroma(collection_name="transcript_db", embedding_function = embedding_function)      
        
        timestampsSuggestion = []
        for topic in topics['function']['topics'] :
            # search vector store 
            docs = db.similarity_search(query = topic, k = 1)     
            # print results
            print("timestamp for " + topic)        
            print(docs[0])

            curTimestamp = 0
            curTimestamp = ConvertTimestampMetadataIntoSeconds(docs[0].metadata)
            print(curTimestamp)            

            # append to array of timestamps
            timestampsSuggestion.append({
                "title" : topic,
                "link"  : f"https://www.youtube.com/embed/{videoID}?fs=1&start={curTimestamp}"
            })

        response_data = {
            "title": videoInfo["videoTitle"],
            "content": summaryOutput,
            "videos": timestampsSuggestion
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


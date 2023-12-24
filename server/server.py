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
from langchain.memory.chat_message_histories import SQLChatMessageHistory
import uuid
import time
from langchain.agents import initialize_agent
# from langchain.tools import DuckDuckGoSearchTool
from langchain.agents import Tool
from langchain.tools import BaseTool

def generate_session_id(string_param):
    # current_time = int(time.time() * 1000)  # Convert current time to milliseconds
    unique_id = uuid.uuid4().hex  # Generate a random UUID as a hex string

    # Combine the current time, string parameter, and random UUID
    # session_id = f"{current_time}_{string_param}_{unique_id}"

    return unique_id


# chat_message_history = SQLChatMessageHistory(
#     session_id="test_session", connection_string="sqlite:///sqlite.db"
# )

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

@app.route("/initializeSession", methods=['POST'])
def InitializeSession():
    try:
        json_data = request.get_json()
        videoInfo = json_data["videoInfo"]
        videoID = videoInfo["videoID"]
        sessionID = generate_session_id(videoID)
        print(f"current session ID {sessionID}")

        response_data = {
            "sessionID": sessionID
        }
        return jsonify(response_data)       
        
    except Exception as e:
        return jsonify({"error" : str(e)})

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
        sessionID = json_data['sessionID']
        videoInfo = json_data['videoInfo']

        print("video ID: ")
        print(videoInfo["videoID"])
        videoID = videoInfo["videoID"]        

        # enable if we need to stop summarization 
        # response_data = {            
        #     "title": videoInfo["videoTitle"],
        #     "content": "dummy summary",
        #     "videos":  [
        #         {"title": "Video 1", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"},
        #         {"title": "Video 2", "link": "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=1500"}
        #     ]
        # }

        # chat_message_history = SQLChatMessageHistory(
        #     session_id = sessionID, connection_string="sqlite:///sqlite.db"
        # )        
        # chat_message_history.add_ai_message(response_data["content"])

        # return jsonify(response_data)

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
            print(docs)            

            curTimestamp = 0
            curTimestamp = ConvertTimestampMetadataIntoSeconds(docs[0].metadata)
            print(curTimestamp)            

            # append to array of timestamps
            timestampsSuggestion.append({
                "title" : topic,
                "link"  : f"https://www.youtube.com/embed/{videoID}?fs=1&start={curTimestamp}&end={curTimestamp}"
            })

        response_data = {
            "title": videoInfo["videoTitle"],
            "content": summaryOutput,
            "videos": timestampsSuggestion
        }

        chat_message_history = SQLChatMessageHistory(
            session_id = sessionID, connection_string="sqlite:///sqlite.db"
        )        
        chat_message_history.add_ai_message(response_data["content"])

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/chatWithContext", methods=['POST'])
def ChatWithContext():
    try:
        json_data = request.get_json()
        
        print(json_data['sessionID'])
        print(json_data['message'])
        # print("request message: " + json_data.message)

        sessionID = json_data['sessionID']
        chat_message_history = SQLChatMessageHistory(
            session_id = sessionID, connection_string="sqlite:///sqlite.db"
        ) 

        print(chat_message_history.messages)

        # print("")
        # print("context search: " + json_data['message'])
        # embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
        # db = Chroma(collection_name="transcript_db", embedding_function = embedding_function)      

        # # search vector store 
        # docs = db.similarity_search(json_data['message']) 

        # # print results        
        # # print(docs[0].page_content)
        # # print(docs[1].page_content)
        # # print(docs[2].page_content)
        
        # videoID = ''
        # print(session)
        # if 'videoID' in session:            
        #     print("video ID is in session!")
        #     videoID = session['videoID']
        #     print (videoID)

        # timestampsSuggestion = []
        # for index, curDoc in enumerate(docs) :
        #     print(f"cur document found: {index}")
        #     print(curDoc.page_content) 
        #     print("\n")
        #     curTimestamp = 0
        #     curTimestamp = ConvertTimestampMetadataIntoSeconds(curDoc.metadata)
        #     timestampsSuggestion.append({
        #         "title" : "topic",
        #         "link"  : f"https://www.youtube.com/embed/{videoID}?fs=1&start={curTimestamp}"
        #     })

        # tools = [search, random_tool, life_tool]

        # # conversational agent memory
        # memory = ConversationBufferWindowMemory(
        #     memory_key='chat_history',
        #     k=3,
        #     return_messages=True
        # )

        # Set up the turbo LLM
        turbo_llm = ChatOpenAI(
            temperature=0.3,
            model_name='gpt-3.5-turbo',
            openai_api_key=ENV_OpenAI_api_key
        )

        # search = DuckDuckGoSearchTool()
        # # defining a single tool
        # tools = [
        #     Tool(
        #         name = "search",
        #         func=search.run,
        #         description="useful for when you need to answer questions about current events. You should ask targeted questions"
        #     )
        # ]
        tools =[]

        # create our agent
        conversational_agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=turbo_llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=chat_message_history
        )

        response_data = {
            "title": "Example Response",
            "content": "This is a sample response with video timestamps.",
            "videos": []
        }
        return jsonify(response_data)
        return jsonify({"response" : "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)})


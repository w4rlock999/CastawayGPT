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
from langchain.chains import LLMChain
from typing import Optional
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
import uuid
import time
from langchain.agents import initialize_agent
# from langchain.tools import DuckDuckGoSearchTool
from langchain.agents import AgentExecutor, Tool, ConversationalChatAgent
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
        db = Chroma.from_documents(persist_directory="./chroma_db", documents = docs, embedding = embedding_function)        

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
        
        db = Chroma(persist_directory="./chroma_db", embedding_function = embedding_function)      
        
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
                "link"  : f"https://www.youtube-nocookie.com/embed/{videoID}?fs=1&start={curTimestamp}&end={curTimestamp}"
            })

        response_data = {
            "title": videoInfo["videoTitle"],
            "content": summaryOutput,
            "videos": timestampsSuggestion
        }

        # response_data = {
        #     "title": videoInfo["videoTitle"],
        #     "content": "content",
        #     "videos": []
        # }

        chat_message_history = SQLChatMessageHistory(
            session_id = sessionID, connection_string="sqlite:///sqlite.db"
        )        
        chat_message_history.add_ai_message(f"this is the summary of the podcast: {response_data['content']}")
        # chat_message_history.add_ai_message(f"this is the summary of the podcast: This podcast discusses the importance of language and how it can be used to create a sense of community, as well as the current multipolar world which has made it more difficult for people to come together. It also mentions the story of a World Champion who wrote a book about how to deal with bullies, which is an inspiration to many people around the world.")        

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/chatWithContext", methods=['POST'])
def ChatWithContext():
    json_data = request.get_json()

    videoOutput = []

    embedding_function = OpenAIEmbeddings(openai_api_key=ENV_OpenAI_api_key)
    db = Chroma(persist_directory="./chroma_db", embedding_function = embedding_function)      
    
    videoInfo = json_data['videoInfo']

    print("video ID: ")
    print(videoInfo["videoID"])
    videoID = videoInfo["videoID"]  

    def TranscriptSimilaritySearch(input=""):
        docs = db.similarity_search(query = input, k = 3)

        output = ""
        for index, doc in enumerate(docs) :            
            output += f"Found Possible Context {index+1} : {doc.page_content} \n\n"
            curTimestamp = 0
            curTimestamp = ConvertTimestampMetadataIntoSeconds(doc.metadata)                      

            # append to array of timestamps
            videoOutput.append({
                "title" : "",
                "link"  : f"https://www.youtube-nocookie.com/embed/{videoID}?fs=1&start={curTimestamp}&end={curTimestamp}"
            })

        return output
    
    transcript_search_tool = Tool(
        name='Transcript Similarity Search',
        func= TranscriptSimilaritySearch,
        description="Useful for when you need additional context from the podcast transcript to answer questions about the podcast content. input should be one topic you need to find out about"
    )

    try:
        
        print(json_data['sessionID'])
        print(json_data['message'])        

        sessionID = json_data['sessionID']
        SQLmessages = SQLChatMessageHistory(
            session_id = sessionID, connection_string="sqlite:///sqlite.db"
        )                   
        print(SQLmessages.messages)
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=SQLmessages, return_messages=True)

        print("building prompt")

        # custom suffix, may needed
        CUSTOM_SUFFIX = """TOOLS
        ------
        Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

        {{tools}}

        {format_instructions}

        CHAT HISTORY
        --------------------
        Here is the previous conversation between you (AI) and the user (Human), use this as a basis for answering next user's input
        {chat_history}

        USER'S INPUT
        --------------------
        Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

        {{{{input}}}}"""

        tools = [transcript_search_tool]                

        prompt = ConversationalChatAgent.create_prompt(
            tools                        
        )
        
        # Set up the LLM
        llm = ChatOpenAI(
            temperature=0.3,
            model_name='gpt-3.5-turbo',
            openai_api_key=ENV_OpenAI_api_key
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ConversationalChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
        )

        input = json_data['message']
        output = agent_chain.run(input=input)
        print(output)
        
        print(SQLmessages.messages)

        response_data = {
            "title": input,
            "content": output,
            "videos": videoOutput
        }
        return jsonify(response_data)
        return jsonify({"response" : "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)})


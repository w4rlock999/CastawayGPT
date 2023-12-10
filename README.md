# CastawayGPT

## Description
CastawayGPT is a tool designed to:
- Summarize YouTube podcasts
- Recommend interesting timestamps for unique topics
- Engage in a chat with context related to the video

## Getting Started
1. Clone the repository.
2. Create a `.env` file under the `server` folder for your OpenAI API key, with `API_KEY` as the variable name.
3. Run `npm install` in the root directory.
4. Install the following Python dependencies using pip:
    ```bash
    pip install langchain chromadb scikit-learn openai flask numpy
    ```
5. Run `npm run dev` in the root folder.
6. Run `flask --app server run` in the `server` folder.

**Note:**
- Currently, the chat with context feature is not yet implemented (as of 10Dec2023).
- OpenAI language models used:
    - GPT-3.5-turbo-0613
    - Text-davinci:003
    - Text-embedding-ada-002-v2
- Summarization is performed using "best representation vectors" to save cost.
- Vector Store utilizes ChromaDB (non-persistent in-memory).
- This repository is a work in progress, and we appreciate the contributions of OpenAI, Langchain, ChromaDB, and Scikit-Learn.

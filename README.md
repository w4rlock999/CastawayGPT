# CastawayGPT

## Description
CastawayGPT is a tool designed to:
- Summarize YouTube podcasts
- Recommend interesting timestamps for unique topics
- Engage in a chat with context related to the video

https://private-user-images.githubusercontent.com/19953728/289363542-17f4fd80-8917-40bb-81be-793e51d2b3a8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIyMTc0MDcsIm5iZiI6MTcwMjIxNzEwNywicGF0aCI6Ii8xOTk1MzcyOC8yODkzNjM1NDItMTdmNGZkODAtODkxNy00MGJiLTgxYmUtNzkzZTUxZDJiM2E4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjEwVDE0MDUwN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTgyMjg5YmUyNWFlYjE3OGEzZjdjNzVmMDNiMmYwNDc0Zjk0N2ZjMzRkODVhYzRkY2NhNDUwM2UxMGUzMWRiYWImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.f71ouQtYdljEiYY_q5vbz2TiGZodLK6nv-GSTuSFu0w




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
- This repository is a work in progress, huge thanks to OpenAI, Langchain, ChromaDB, and Scikit-Learn.

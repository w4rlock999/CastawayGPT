import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { YoutubeTranscript, TranscriptResponse } from 'youtube-transcript';
import { Chroma } from "langchain/vectorstores/chroma";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";


interface HMSTimestamp {
    hours: number;
    minutes: number;
    seconds: number;
}

interface TranscriptObject {
    timestamp: HMSTimestamp;
    text: string;
}

function millisecondsToHMS(milliseconds : number): HMSTimestamp {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
  
    const remainingSeconds = seconds % 60;
    const remainingMinutes = minutes % 60;
  
    return {
      "hours": hours,
      "minutes": remainingMinutes,
      "seconds": remainingSeconds
    };
  }

function splitTranscript(objectArray : TranscriptResponse[], splitInto : number): TranscriptObject[] {

    let arrayLength = objectArray.length
    var splittedTranscript : TranscriptObject[] = [];

    for(let i = 0; i < arrayLength; i+=splitInto) {
        var curEnd = Math.min(i+splitInto,arrayLength);
        var curOffset = objectArray[i].offset;
        var combinedText = objectArray.slice(i, curEnd).map(obj => obj.text).join(" ");

        var curHMS : HMSTimestamp = millisecondsToHMS(curOffset);        
        // console.log("offset: " + curOffset + " text: " + combinedText)

        var curSplittedTranscript : TranscriptObject = {
            timestamp: curHMS,
            text: combinedText
        };    
        splittedTranscript.push(curSplittedTranscript);
    }

    return splittedTranscript;
}

export async function POST() {
    var objectArray: TranscriptResponse[] = await YoutubeTranscript.fetchTranscript('https://youtu.be/vnVfxu530AM');
    
    var splittedTranscript : TranscriptObject[] = await splitTranscript(objectArray, 5);
    // console.log(splittedTranscript);
    // splittedTranscript.forEach(transcript => {
    //     console.log(transcript.text);
    // });

    // const subsetDocument = splittedTranscript.slice(0, 5); // Extract the first five members
    const transcriptDocuments = splittedTranscript.map(obj => (
        {                  
          page_content : obj.text,
          metadata : {timestamp: obj.timestamp}                          
        }
      ));

    console.log(transcriptDocuments)  

    return new Response("fetched youtube");


    const response = await fetch('http://127.0.0.1:5000/addEmbeddingToChroma', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',        
      },
      body: JSON.stringify(transcriptDocuments),
    });

    if (response.ok) {
      const result = await response.json();
      // Response.status(200).json({ message: 'Data sent successfully', result });
      console.log(JSON.stringify(result))
      return new Response(JSON.stringify(result))
    } else {      
      return Response.error()

      // Response.status(500).json({ error: 'Failed to send data' });
    }
  
    return new Response("fetched youtube");


    // /* Load in the file we want to do question answering over */
    // const text = fs.readFileSync("data/state_of_the_union.txt", "utf8");

    // /* Split the text into chunks */
    // const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    // const docs = await textSplitter.createDocuments([text]);

    // // console.log(docs)

    // const vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    //   collectionName: "state_of_the_union",
    // });
    // const vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    //   collectionName: "state-of-the-union-document",
    //   url: "http://localhost:8000", // Optional, will default to this value
    //   collectionMetadata: {
    //     "hnsw:space": "cosine",
    //   }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
    // });


    // Pseudo code
    // first make splitter and convert the offset into youtube time format [DONE]
    // create embedding of the splitted document [failed on JS, proceed with python]
    // store into chroma [procedd with python]
    // make SINGLE chat query into the document
}   
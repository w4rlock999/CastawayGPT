import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { YoutubeTranscript, TranscriptResponse } from 'youtube-transcript';
import { Chroma } from "langchain/vectorstores/chroma";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

interface Payload {
  youtubeLink : string
}

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

function regroupTranscript(objectArray : TranscriptResponse[], groupMembers : number): TranscriptObject[] {

  let arrayLength = objectArray.length
  var regroupedTranscript : TranscriptObject[] = [];

  for(let i = 0; i < arrayLength; i+=groupMembers) {
      var curEnd = Math.min(i+groupMembers,arrayLength);
      var curOffset = objectArray[i].offset;
      var combinedText = objectArray.slice(i, curEnd).map(obj => obj.text).join(" ");

      var curHMS : HMSTimestamp = millisecondsToHMS(curOffset);        
      // console.log("offset: " + curOffset + " text: " + combinedText)

      var curTranscriptGroup : TranscriptObject = {
          timestamp: curHMS,
          text: combinedText
      };    
      regroupedTranscript.push(curTranscriptGroup)
  }

  return regroupedTranscript;
}

export async function POST(request: Request) {

    const data : Payload = await request.json()  

    if (data.youtubeLink != "") {
      // var objectArray: TranscriptResponse[] = await YoutubeTranscript.fetchTranscript(data.youtubeLink);
    
      // var splittedTranscript : TranscriptObject[] = await regroupTranscript(objectArray, 10);    
      
      // const subsetDocument = splittedTranscript.slice(0, 25); // Extract the first five members
      // const transcriptDocuments = subsetDocument.map(obj => (
      //     {                  
      //       page_content : obj.text,
      //       metadata : {timestamp: obj.timestamp}                          
      //     }
      //   ));
  
      // console.log(transcriptDocuments)  
      // console.log(transcriptDocuments.length)  
    
      // var response = await fetch('http://127.0.0.1:5000/addEmbeddingToChroma', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',        
      //   },
      //   body: JSON.stringify(transcriptDocuments),
      // });
  
      // response = response && await fetch('http://127.0.0.1:5000/chatWithContext', {
      var response = await fetch('http://127.0.0.1:5000/chatWithContext', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',        
        },
        body: JSON.stringify({
          message: "initialize chat with video summary"
        }),
      });

      if (response.ok) {
        const result = await response.json()
        console.log(JSON.stringify(result))
        return new Response(JSON.stringify(result))
      } else {      
        return Response.error()      
      }
        
      return new Response("Initialized success!");
    }else{
      console.log("Link not valid!")
      return Response.error()   
    }

}
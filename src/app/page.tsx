'use client';

import Image from 'next/image'
import styles from './page.module.css'
import { PromptTemplate } from "langchain/prompts";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { Chroma } from "langchain/vectorstores/chroma";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { useEffect, useState } from 'react';
import { Console } from 'console';

interface videoTimestamp {
  title: string;
  link : string;
}

interface chatResponse {
  title : string;
  content : string;
  videos : videoTimestamp[];
}

function ChatResponse(props) {
  
  return (
    <div className = {styles.responseContainer}>
      <h1 className = {styles.responseTitle}>
        {props.curChatResponse.title}
      </h1>
      <br></br>
      <br></br>            
      <p className = {styles.responseBody}>
        {props.curChatResponse.content}
      </p>            
      <br></br>            
      <br></br>    
      <div className = {styles.videosContainer}>
        {props.curChatResponse.videos?.map((curVideo, index) => (
          <div className = {styles.videoContainer}>
          <iframe width="270" height="180"
            src={curVideo.link} frameborder="0" allow="fullscreen">
          </iframe>
          <p className = {styles.videoText}>
            {curVideo.title}
          </p>
        </div> 
        ))}               
      </div>                                   
    </div>
  )
}

export default function Home() {

  const curChatResponse : chatResponse = {
    title: "Elon Musk: War, AI, Aliens, Politics, Physics, Video Games, and Humanity | Lex Fridman Podcast #400",
    content: "Lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet dolor sit amet amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum dolor sit amet lorem ipsum",
    videos: [
      {
        title: "2:30 Lorem ipsum dolor sit amet",
        link: "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"
      },{
        title: "2:30 Lorem ipsum dolor sit amet",
        link: "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"
      },{
        title: "2:30 Lorem ipsum dolor sit amet",
        link: "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"
      },{
        title: "2:30 Lorem ipsum dolor sit amet",
        link: "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"
      },{
        title: "2:30 Lorem ipsum dolor sit amet",
        link: "https://www.youtube.com/embed/JN3KPFbWCy8?fs=1&start=500"
      },
    ]
  }

  const curChatResponsesDummiesThree : chatResponse[] = [curChatResponse,curChatResponse,curChatResponse]
  const curChatResponsesDummiesOne : chatResponse[] = [curChatResponse]

  const [youtubeLink, setYoutubeLink] = useState('')  
  const [promptSearch, setPromptSearch] = useState('')
  const [landingMode, setLandingMode] = useState(true)
  const [chatsHistory, setChatsHistory] = useState([])
  const [chatMessage, setChatMessage] = useState('')
  const [responses, setResponses] = useState<chatResponse[]>([])
  // setArrayOfObjects((prevArray) => [...prevArray, newObject]); //to add new object to the array, use this 
  
  const runButtonOnClickHandler = async () => {    

    let response = await fetch('/api/initializeWithLink', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',        
      },
      body: JSON.stringify({youtubeLink})
    })         
    
    if (response.ok) {
      const responseData : chatResponse = await response.json()
      // console.log(responseData)
      setResponses((prevArray) => [...prevArray, responseData]) 
      setLandingMode(false)
    }
  }  

  const linkOnChangeHandler = (e) => {
    setYoutubeLink(e.target.value)
  }

  const chatOnChangeHandler = (e) => {
    setChatMessage(e.target.value)
  }  

  const chatButtonOnClickHandler = async () => {    

    var curChatMessage = chatMessage
    setChatMessage("")
    console.log(curChatMessage)

    const bodyPayload = {
      chatMessage : curChatMessage
    }

    let response = await fetch('/api/chatWithRAG', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',        
      },
      body: JSON.stringify(bodyPayload)
    })   

    if (response.ok) {
      const responseData : chatResponse = await response.json()
      // console.log(responseData)
      setResponses((prevArray) => [...prevArray, responseData])       
    }        
  }

  return (
    <div className = {styles.container}>

      {landingMode == true && 
        <div className = {styles.landingPageContainer}>
          <h2 className = {styles.logo}>
            CastawayGPT
          </h2>
          <input className = {styles.linkInput} type='text' placeholder='Paste your youtube link here' value={youtubeLink} onChange={linkOnChangeHandler}>
          </input>
          <button className = {styles.runButton} onClick={runButtonOnClickHandler}>
            Start Chat
          </button>      
        </div>
      }

      {landingMode == false && 
        <div className = {styles.chatPageContainer}>          
          {responses.map((response, index) => (
            <ChatResponse curChatResponse = {response}/>    
          ))}          
          <div className = {styles.bottomSpacer}>
          </div>

          <div className = {styles.chatInputAndButtonContainer}>          
            <input className = {styles.chatInput} type='text' placeholder='Ask anything here!' value={chatMessage} onChange={chatOnChangeHandler}>
            </input>                    
            <button className = {styles.sendButton} onClick={chatButtonOnClickHandler}>       
            ▲       
            </button>        
          </div>        
        </div> 
      } 

      <div className = {styles.circle1}>        
      </div>
      <div className = {styles.circle2}>        
      </div>
      {/* <div className = {styles.promptContainer}>
        <input className = {styles.inputPrompt} type='text' placeholder='Prompt search here' value={promptSearch} onChange={promptOnChangeHandler}>
        </input>
        <button className = {styles.searchButton} onClick={searchButtonOnClickHandler}>
          Search
        </button>      
      </div> */}
    </div>
  )
}

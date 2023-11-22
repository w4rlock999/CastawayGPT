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

export default function Home() {

  const [youtubeLink, setYoutubeLink] = useState('')
  const [chatMessage, setChatMessage] = useState('')
  const [promptSearch, setPromptSearch] = useState('')
  const [landingMode, setLandingMode] = useState(true)
  const [chatsHistory, setChatsHistory] = useState([])

  const runButtonOnClickHandler = async () => {    
    setLandingMode(false)

    let response = await fetch('/api/initializeWithLink', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',        
      },
      body: JSON.stringify({youtubeLink})
    })   
        
  }

  const linkOnChangeHandler = (e) => {
    setYoutubeLink(e.target.value)
  }

  const chatButtonOnClickHandler = async () => {    
    // let response = await fetch('/api/searchWithPrompt', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',        
    //   },
    //   body: JSON.stringify({promptSearch})
    // })   
    setLandingMode(true)        
  }

  const chatOnChangeHandler = (e) => {
    setChatMessage(e.target.value)
  }

  return (
    <div className = {styles.container}>

      {landingMode == true && 
        <div className = {styles.linkContainer}>
          <input className = {styles.linkInput} type='text' placeholder='Paste your youtube link here' value={youtubeLink} onChange={linkOnChangeHandler}>
          </input>
          <button className = {styles.runButton} onClick={runButtonOnClickHandler}>
            Start Chat
          </button>      
        </div>
      }

      {landingMode == false &&              
        <div className = {styles.chatPageContainer}>
          <div className = {styles.chatInputAndButtonContainer}>          
            <input className = {styles.chatInput} type='text' placeholder='Ask anything here!' value={chatMessage} onChange={chatOnChangeHandler}>
            </input>                    
            <button className = {styles.sendButton} onClick={chatButtonOnClickHandler}>       
            ▲       
            </button>        
            {/* <button className = {styles.sendButton}>            
            </button>  */}
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

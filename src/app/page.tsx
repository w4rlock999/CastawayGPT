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
  const [promptSearch, setPromptSearch] = useState('')

  const runButtonOnClickHandler = async () => {    
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

  const searchButtonOnClickHandler = async () => {    
    let response = await fetch('/api/searchWithPrompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',        
      },
      body: JSON.stringify({promptSearch})
    })   
        
  }

  const promptOnChangeHandler = (e) => {
    setPromptSearch(e.target.value)
  }

  return (
    <div className = {styles.container}>
      <div className = {styles.linkContainer}>
        <input className = {styles.inputLink} type='text' placeholder='Your youtube link here' value={youtubeLink} onChange={linkOnChangeHandler}>
        </input>
        <button className = {styles.runButton} onClick={runButtonOnClickHandler}>
          Start Chat
        </button>      
      </div>
      <div className = {styles.promptContainer}>
        <input className = {styles.inputPrompt} type='text' placeholder='Prompt search here' value={promptSearch} onChange={promptOnChangeHandler}>
        </input>
        <button className = {styles.searchButton} onClick={searchButtonOnClickHandler}>
          Search
        </button>      
      </div>
    </div>
  )
}

'use client';

import Image from 'next/image'
import styles from './page.module.css'
import { PromptTemplate } from "langchain/prompts";
import { ChatOpenAI } from "langchain/chat_models/openai";

const model = new ChatOpenAI({  
  openAIApiKey: "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",  
});

const promptTemplate = PromptTemplate.fromTemplate(
  "Tell me a joke about {topic}"
);

export default function Home() {

  const runButtonOnClickHandler = async () => {
    console.log("Hello World!")

    const chain = promptTemplate.pipe(model);
    const result = await chain.invoke({ topic: "bears" });
    console.log(result);
  }

  return (
    <div className = {styles.container}>
      <button className = {styles.runButton} onClick={runButtonOnClickHandler}>
        RUN
      </button>
    </div>
  )
}

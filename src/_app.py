import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.agents import AgentExecutor
from langchain.vectorstores import FAISS

import streamlit as st

load_dotenv(dotenv_path="./")


def load_and_summarize_doc(url):
    """Loads text from a url and summarizes it using the summarization chain"""
    loader = WebBaseLoader(url)
    docs = loader.load()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return llm, docs, summary


def create_vector_db(llm, url, docs):
    """Creates a vector database from a list of documents"""
    # we will create a local (vector) database
    persist_directory = f"../vector_db/{url.split('/')[-2]}"
    embeddings = OpenAIEmbeddings()

    if not os.path.exists(persist_directory):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        documents = splitter.split_documents(docs)

        # create the vector db
        vectordb = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        vectordb.save_local(persist_directory)
    else:
        # if the db already exists, load it
        vectordb = FAISS.load_local(persist_directory, embeddings)

    # create a retriever
    doc_retriever = vectordb.as_retriever()

    doc_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=doc_retriever
    )
    return doc_qa


def create_chat_agent(llm, doc_qa):
    """Creates a chat agent that can answer questions about the document"""

    tools = [
        Tool(
            name="Document QA Agent",
            func=doc_qa.run,
            description="useful for when you need to answer questions about a document. Input should be a fully formed question.",
        )
    ]

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, memory=memory
    )

    return agent_chain


def main():
    st.set_page_config(layout="wide")
    st.title("Talk to a document")
    st.markdown(
        "This app lets you summarize a document and also ask questions about the document."
    )
    url = st.text_area("Enter a url that contains some text:", height=25)

    if url:
        llm, docs, summary = load_and_summarize_doc(url)
        st.markdown("### Summary:")
        st.markdown(summary)

        # document retriever
        doc_qa = create_vector_db(llm, url, docs)

        # create LLMChain agent
        agent_chain = create_chat_agent(llm, doc_qa)

        question = st.text_area("Ask a question about the document:", height=50)
        if question:
            answer = agent_chain.run(input=question)
            st.markdown("#### Answer:")
            st.markdown(answer)


if __name__ == "__main__":
    main()

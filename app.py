import streamlit as st
import json
import os
import openai
import pinecone
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI




# Environment variables 
load_dotenv("key.env")
openai.api_key = os.getenv('OPENAI_API_KEY')


# Index name
index_name = "smartpreppeprr"
pc = Pinecone(api_key="pcsk_4oqbPN_EqGQuGKCHfdVH4osGkQk48wHcQ1kzx77QfM7fm7TbA6bjwfxwU9FAABMeLNWhDM")  


# Creating pinecone index 
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
pinecone_index = pc.Index(index_name)




# Embeddings and vector store for retrieval
embeddings = OpenAIEmbeddings()
text_key = "text"
vectorstore = PineconeVectorStore(pinecone_index, embeddings, text_key)



# Storing conversation in .txt file
def store_conversation(question, response):
    with open("conversation_history.txt", "a") as file:
        file.write(f"Q: {question}\nA: {response}\n\n")




def generate_openai_prompt(context, message):
    prompt = f"""Given a chat history and the latest user question {message}
    which might reference context in the chat history {context}, 
    formulate a standalone question which can be understood 
    without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    
    Additionally:
    
    If the user greets you with phrases like "hi", "hello", "hey", "good morning", "good evening", "how are you", or any other typical greeting, 
    your response should always and only be "hi". Do not include any additional words, phrases.
      
      """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()
    




# Rag-Pipeline
def ask_openai_with_rag(context, message):
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18", openai_api_key=openai.api_key),
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    prompt = generate_openai_prompt(context, message)
    response = qa.invoke({"query": prompt})
    return response.get("result", "Error: No result received.")

# Storing the conversation
def get_last_5_conversations():
    try:
        with open("conversation_history.txt", "r") as file:
            lines = file.readlines()

        
        conversations = []
        for i in range(0, len(lines), 3):  
            if i + 2 < len(lines):  
                question = lines[i].strip()
                answer = lines[i + 1].strip()
                conversations.append(f"{question}\n{answer}")

        
        return "\n\n".join(conversations[-5:])
    except FileNotFoundError:
        return "No conversation history found."



# Streamline
st.set_page_config(page_title=" Chatbot", layout="wide")
st.title(" AI Chatbot")
st.markdown(
    """Hi, I am your AI assistant. I can solve and answer all your problems and queries related to UPSC. How can I help you?"""
)





#Chat History
st.sidebar.title("Chat History")
st.sidebar.markdown("Here are the last 5 conversations:")
st.sidebar.text(get_last_5_conversations())

user_question = st.text_input("Ask me anything:")
if st.button("Submit"):
    if user_question.strip():
        # Retrieve last 5 conversations as context
        context = get_last_5_conversations()
        print(context)

        # Generate and display response
        with st.spinner("Generating response..."):
            chatbot_response = ask_openai_with_rag(context, user_question)

        st.markdown(f"  You:   {user_question}")
        st.markdown(f"  Chatbot:   {chatbot_response}")

        # Store conversation
        store_conversation(user_question, chatbot_response)
    else:
        st.error("Please enter a valid question.")







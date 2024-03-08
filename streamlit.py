import os
import pickle
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "./storage"
INDEX_FILE = os.path.join(PERSIST_DIR, "index.pkl")
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as file:
        index = pickle.load(file)
else:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(INDEX_FILE, "wb") as file:
        pickle.dump(index, file)
listing_history=[]
st.write("Welcome to USC GOLD chatbot")
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)
if 'listing_history' not in st.session_state:
    st.session_state['listing_history'] = []

user_query = st.text_input("Enter the query:", key="user_query")

if st.button("Answer"):
    if user_query.lower() == "exit":
        st.stop()
    else:
        output = query_engine.query(user_query)
      
        st.session_state['listing_history'].append((user_query,output))
        st.write(output.response)

st.write("Previous Chats:")
for question, answer in st.session_state['listing_history']:
    st.write(f"Question: {question}")
    st.write(f"Answer: {answer}")
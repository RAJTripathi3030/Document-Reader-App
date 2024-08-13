import os 
import time 
import streamlit as st 
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
# from io import BytesIO 
from langchain.docstore.document import Document
# For Splitting Text Into Chunks 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# Vector Store DB
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# For Embeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# For the environment variables
load_dotenv()

# Inputting the API keys from the environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title = "Document Reader Bot", page_icon = "🤖")
st.title("Document Q&A Chatbot")
with st.expander(label = "***Instructions For Using The Q&A Chatbot :***"):
    st.markdown("""
            
            :blue[1. Before starting to chat with the bot, please insert some documents.]
            
            :blue[2. After insertion, please press the :red[***Process the document***] button.]
            
            :blue[3. Please provide very clear instructions.] 
            
            :blue-background[Also if the app stopped working and raises errors then maybe I ran out of my token limit :)]
            """)

# Specifying the Language Model to Use
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Gemma2-9b-It")

prompt = ChatPromptTemplate.from_template(
"""
You are a very friendly bot, and you answer the questions based on the provided context.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
            
def create_embeddings(text):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        final_documents = st.session_state.text_splitter.split_text(text)
        st.session_state.final_documents = [Document(page_content=chunk) for chunk in final_documents]
        # Now we create a vector store where we need to pass our final_document and embeddings in the from_documents() method.  
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
# Creating a sidebar where the user will input the document/s and they will all be parsed and concatinated in text
text = ""
with st.sidebar:
    st.subheader("Select your files :)")
    # File Upload
    input_file = st.file_uploader("Upload a PDF File", type = 'pdf', accept_multiple_files = True)
    
    for pdf in input_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # At this point "text" string has everything that was in the document/s in it.
    # Press the processing button that calls create_embeddings() 
    if st.button("Process the document"):
        with st.spinner("Processing..."):
            create_embeddings(text)
        st.balloons()
        st.success("Processed Successfully!")

# Here the user enters the prompt
if prompt1 := st.chat_input(placeholder = "Talk to me"):
    st.session_state.messages.append({"role": "user", "content": prompt1})
    with st.chat_message("user"):
        st.write(prompt1)
        
# 
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    
    # Display assistant's response
    with st.chat_message("assistant"):
        st.write(response['answer'])

    # # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("--------------------------------") 
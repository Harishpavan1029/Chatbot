import streamlit as st
from Chatbot import process_document, generate_response, initialize_vector_store
import os

# Streamlit app configuration
st.set_page_config(page_title="Document-Based Chatbot", page_icon="📚")
st.title("Document-Based Chatbot")
st.write("Upload a document (.txt or .pdf) and ask questions!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize ChromaDB
initialize_vector_store()

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file:
    # Save the uploaded file
    file_path = os.path.join("documents", uploaded_file.name)
    os.makedirs("documents", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the document
    try:
        process_document(file_path)
        st.success(f"Document '{uploaded_file.name}' processed successfully!")
    except Exception as e:
        st.error(f"Error processing document: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user message
user_input = st.chat_input("Ask a question about the document...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display bot response
    with st.chat_message("assistant"):
        response = generate_response(user_input, st.session_state.messages[:-1])
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #d1e7ff; }
    .stChatMessage.assistant { background-color: #e6f3e6; }
    </style>
""", unsafe_allow_html=True)
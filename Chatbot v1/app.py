import streamlit as st
from Chatbot import generate_response  # Import the function from the previous step

# Streamlit app configuration
st.title("My Chatbot")
st.write("Ask me anything!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user message
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display bot response
    with st.chat_message("assistant"):
        response = generate_response(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


st.title("Mini Project 2: Streamlit Chatbot")

# TODO: Replace with your actual OpenAI API key
# client = OpenAI(api_key='OPENAI_API_KEY')
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    conversation = ""
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        conversation += f"{role}: {content}\n"
    return conversation

# Check for existing session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # Append user message to messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages
        )
        assistant_reply = response.choices[0].message.content
        st.markdown(assistant_reply)

    # Append AI response to messages
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
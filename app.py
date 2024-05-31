import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("EMILDAI CHATBOT")

# Function to update the knowledge base
#def update_kb():
#    create_vector_db()
#    st.write("Knowledge base updated!")

# Button to update the knowledge base
#if st.button("Update the Knowledge base"):
#    update_kb()

# Text input for the question
question = st.text_input("Question: ")

# Function to get the response and display it
def get_and_display_response(question):
    chain = get_qa_chain()
    response = chain(question)

    #st.header("Answer")
    st.write(response["result"])

# Button to submit the question
if st.button("Submit"):
    if question:
        get_and_display_response(question)

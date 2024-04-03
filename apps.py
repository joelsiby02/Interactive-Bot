import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
import os
import langchain
# Load environment variables from .env file
load_dotenv()

# Check if GOOGLE_API_KEY is set in the environment, if not prompt the user
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.text_input("Provide your Google API Key", type="password")

# Create a Google Generative AI chat instance
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define the conversation prompt template
template = """
"Start a conversation with the user, focusing on their interests and preferences. Actively listen to their input to identify their interests and engage them accordingly. If there's something you're unfamiliar with, be honest and express it as "I'm not sure." Feel free to provide additional information or suggestions to keep the conversation flowing.

Current conversation:
{history}
Human: {input}
AI Assistant:
"""

# Define the conversation chain with the prompt template
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

# Streamlit UI
st.title("Conversation AI")

# User input text box
user_input = st.text_input("User Input", "")

# Button to start the conversation
if st.button("Start Conversation"):
    # Get the AI's response
    ai_response = conversation.predict(input=user_input)
    # Display AI's response
    st.write("AI Assistant:", ai_response)

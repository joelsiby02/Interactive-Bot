import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables from .env file
load_dotenv()


# Function to get user input for Google API key
def get_google_api_key():
    st.sidebar.subheader("Enter your Google API Key")
    google_api_key = st.sidebar.text_input("Google API Key", type="password")
    return google_api_key

# Function to configure Google API key
def configure_google_api_key(google_api_key):
    # You can configure the API key here
    pass


def built_app(google_api_key):
    # Streamlit UI
    st.title("What's Reminder")

    # Check if GOOGLE_API_KEY is set in the environment, if not prompt the user
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    # Create a Google Generative AI chat instance
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # Define the conversation prompt template
    template = """
    ""Given a chat message, extract relevant reminder details such as dates, times, events, and tasks. Combine this information to create a list of tasks for a todo list based on user casual messages.
    the output format should be strictly maintained, i will provide an example how i prefer the output to be
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

    # User input text box
    user_input = st.text_input("User Input", "")

    # Button to start the conversation
    if st.button("Get reminders"):
        # Get the AI's response
        ai_response = conversation.predict(input=user_input)
        # Display AI's response
        st.write("AI Assistant:", ai_response)

# Main function
def main():
    # Sidebar for entering API key
    google_api_key = get_google_api_key()

    # Build the app with the provided API key
    built_app(google_api_key)

if __name__ == "__main__":
    main()
